import requests
import os
import json
import re
import time
from pathlib import Path

# Configuration
GITHUB_REPO = "Qiratsumra/ai-book"
GITHUB_BRANCH = "main"
RAG_API_URL = "http://localhost:8000"

def fetch_github_tree(owner, repo, branch="main"):
    """Fetch complete file tree from GitHub repository."""
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    
    print(f"Fetching repository tree from {url}...")
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return []
    
    tree = response.json()
    
    # Filter for markdown files in docs directory
    markdown_files = [
        item for item in tree.get('tree', [])
        if item['type'] == 'blob' 
        and item['path'].startswith('docs/')
        and (item['path'].endswith('.md') or item['path'].endswith('.mdx'))
    ]
    
    return markdown_files

def download_file_content(owner, repo, path, branch="main"):
    """Download file content from GitHub."""
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            print(f"  ✗ Failed to download {path}: {response.status_code}")
            return None
    except Exception as e:
        print(f"  ✗ Error downloading {path}: {e}")
        return None

def clean_markdown_content(content):
    """Clean markdown content for better RAG performance."""
    if not content:
        return ""
    
    # Remove frontmatter (YAML between --- markers)
    content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL | re.MULTILINE)
    
    # Remove HTML comments
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    
    # Remove import statements
    content = re.sub(r'^import\s+.?from\s+[\'"].?[\'"];?\s*$', '', content, flags=re.MULTILINE)
    
    # Remove JSX/MDX components that might be empty
    content = re.sub(r'<[A-Z][a-zA-Z]\s/>', '', content)
    
    # Remove excessive newlines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Remove empty links
    content = re.sub(r'\[([^\]]+)\]\(\)', r'\1', content)
    
    # Strip code blocks that are just imports
    content = re.sub(r'[a-z]*\nimport.*?\n', '', content, flags=re.DOTALL)
    
    return content.strip()

def extract_title_from_content(content, filename, path):
    """Extract title from markdown content."""
    # Try to find first # heading
    match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    
    # Try to extract from path structure
    if '/' in path:
        parts = path.split('/')
        if len(parts) > 1:
            return parts[-2].replace('-', ' ').title() + ": " + filename.replace('.md', '').replace('.mdx', '').replace('-', ' ').title()
    
    # Fallback to filename
    return filename.replace('.md', '').replace('.mdx', '').replace('-', ' ').title()

def categorize_by_path(path):
    """Extract category from file path."""
    parts = path.split('/')
    if len(parts) > 2:
        return parts[1]  # First subdirectory after 'docs/'
    return "General"

def split_into_chunks(content, title, path, category, max_chunk_size=2000):
    """Split large documents into smaller chunks."""
    chunks = []
    
    if len(content) <= max_chunk_size:
        return [{
            'content': content,
            'title': title,
            'path': path,
            'category': category,
            'chunk_index': 0,
            'total_chunks': 1
        }]
    
    # Split by major headings (## or ###)
    sections = re.split(r'\n(?=#{2,3}\s)', content)
    
    current_chunk = ""
    chunk_count = 0
    
    for section in sections:
        # If adding this section would exceed limit, save current chunk
        if len(current_chunk) + len(section) > max_chunk_size and current_chunk:
            chunks.append({
                'content': current_chunk.strip(),
                'title': title,
                'path': path,
                'category': category,
                'chunk_index': chunk_count,
                'total_chunks': 0  # Will update later
            })
            chunk_count += 1
            current_chunk = section
        else:
            current_chunk += "\n" + section if current_chunk else section
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append({
            'content': current_chunk.strip(),
            'title': title,
            'path': path,
            'category': category,
            'chunk_index': chunk_count,
            'total_chunks': 0
        })
    
    # Update total chunks count
    total = len(chunks)
    for chunk in chunks:
        chunk['total_chunks'] = total
    
    return chunks

def upload_bulk_documents(documents, batch_size=15):
    """Upload documents in batches."""
    total_uploaded = 0
    total_failed = 0
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        
        try:
            response = requests.post(
                f"{RAG_API_URL}/upload-documents-bulk",
                json={"documents": batch},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code == 200:
                total_uploaded += len(batch)
                print(f"  ✓ Batch {i//batch_size + 1}: Uploaded {len(batch)} documents")
            else:
                total_failed += len(batch)
                print(f"  ✗ Batch {i//batch_size + 1}: Failed - {response.text[:100]}")
        except Exception as e:
            total_failed += len(batch)
            print(f"  ✗ Batch {i//batch_size + 1}: Error - {str(e)[:100]}")
        
        # Small delay to avoid overwhelming the server
        time.sleep(1)
    
    return total_uploaded, total_failed

def main():
    print("=" * 70)
    print("AI Book Documentation Indexer for RAG Chatbot")
    print("Repository: https://github.com/Kinzaqirat/robotics-book")
    print("=" * 70)
    print()
    
    # Extract owner and repo from GITHUB_REPO
    owner, repo = GITHUB_REPO.split('/')
    
    # Step 1: Fetch repository tree
    print("[1/5] Fetching repository structure from GitHub...")
    files = fetch_github_tree(owner, repo, GITHUB_BRANCH)
    print(f"✓ Found {len(files)} markdown files in docs/")
    print()
    
    if not files:
        print("✗ No files found. Please check the repository URL and branch.")
        return
    
    # Step 2: Download and process files
    print("[2/5] Downloading and processing documentation...")
    all_documents = []
    doc_id = 1
    stats = {'categories': {}, 'files_processed': 0, 'files_failed': 0}
    
    for file_info in files:
        path = file_info['path']
        filename = os.path.basename(path)
        
        # Download content
        content = download_file_content(owner, repo, path, GITHUB_BRANCH)
        
        if not content:
            stats['files_failed'] += 1
            continue
        
        # Clean and process
        cleaned_content = clean_markdown_content(content)
        
        if not cleaned_content or len(cleaned_content) < 50:
            print(f"  ⊘ Skipped {path} (too short or empty)")
            continue
        
        # Extract metadata
        title = extract_title_from_content(cleaned_content, filename, path)
        category = categorize_by_path(path)
        
        # Track categories
        stats['categories'][category] = stats['categories'].get(category, 0) + 1
        
        # Split into chunks if needed
        chunks = split_into_chunks(cleaned_content, title, path, category)
        
        # Prepare documents
        for chunk in chunks:
            all_documents.append({
                "id": doc_id,
                "content": chunk['content'],
                "metadata": {
                    "title": chunk['title'],
                    "path": chunk['path'],
                    "category": chunk['category'],
                    "chunk_index": chunk['chunk_index'],
                    "total_chunks": chunk['total_chunks'],
                    "source": "ai-book",
                    "repo": GITHUB_REPO,
                    "file_name": filename
                }
            })
            doc_id += 1
        
        stats['files_processed'] += 1
        print(f"  ✓ Processed: {path} → {len(chunks)} chunk(s)")
    
    print()
    print(f"✓ Processed {stats['files_processed']} files")
    print(f"✓ Created {len(all_documents)} document chunks")
    print(f"✓ Categories found: {', '.join(stats['categories'].keys())}")
    print()
    
    # Step 3: Upload to RAG
    print("[3/5] Uploading to RAG chatbot...")
    uploaded, failed = upload_bulk_documents(all_documents)
    print()
    
    # Step 4: Save metadata
    print("[4/5] Saving metadata...")
    metadata = {
        'repository': GITHUB_REPO,
        'branch': GITHUB_BRANCH,
        'total_files': len(files),
        'files_processed': stats['files_processed'],
        'files_failed': stats['files_failed'],
        'total_chunks': len(all_documents),
        'uploaded': uploaded,
        'failed': failed,
        'categories': stats['categories'],
        'indexed_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('ai_book_index_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to: ai_book_index_metadata.json")
    print()
    
    # Step 5: Summary
    print("[5/5] Indexing Complete!")
    print("=" * 70)
    print(f"✓ Successfully uploaded: {uploaded} documents")
    if failed > 0:
        print(f"✗ Failed to upload: {failed} documents")
    print()
    print("Categories indexed:")
    for cat, count in stats['categories'].items():
        print(f"  • {cat}: {count} files")
    print("=" * 70)
    print()
    print("🎉 Your AI Book documentation is now searchable!")
    print()
    print("Test your chatbot:")
    print('  curl -X POST "http://localhost:8000/chat" \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"query": "What is artificial intelligence?", "top_k": 5}\'')
    print()
    print("Or with streaming:")
    print('  curl -X POST "http://localhost:8000/chat" \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"query": "Explain machine learning", "top_k": 5, "stream": true}\'')

if __name__ == "_main_":
    main()