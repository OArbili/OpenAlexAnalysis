import os
import time
import json
import requests

def fetch_and_save_articles(resume_from_page=None):
    """
    Fetch all articles matching:
      - publication_year=2024
      - US-affiliated (institutions.country_code=us)
      - cited_by_count>4 (≥ 5 citations)
    Then save them to a JSON Lines file in a new folder named 'articles_2023'.
    
    Args:
        resume_from_page: If provided, resume from this page number.
                         Will load the cursor from the last completed page.
    """
    
    # 1. Create the folder if it doesn't exist
    folder_name = "articles_2024"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # 2. Prepare the output file paths (JSON Lines)
    # Main output file that will contain all articles
    output_file = os.path.join(folder_name, "all_articles.jsonl")
    # Folder for iteration files
    iterations_folder = os.path.join(folder_name, "iterations")
    if not os.path.exists(iterations_folder):
        os.makedirs(iterations_folder)
    
    # 3. OpenAlex base URL
    base_url = "https://api.openalex.org/works"
    
    # 4. Initial query parameters:
    #    - remove type:journal-article if you want *all* types of works
    params = {
        "filter": "publication_year:2024,institutions.country_code:us,cited_by_count:>3",
        "select": "display_name,topics,publication_date", 
        "per-page": 200,  # Maximum 200 articles per page
        "cursor": "*"     # Starting cursor for pagination
    }
    
    # Track the total count of articles we get
    total_downloaded = 0
    page_count = 0
    
    # Check if we should resume from a specific page
    if resume_from_page:
        # Find the cursor for the page we want to resume from
        last_page = resume_from_page - 1
        cursor_file = os.path.join(iterations_folder, f"cursor_after_page_{last_page}.txt")
        
        if os.path.exists(cursor_file):
            with open(cursor_file, "r") as f:
                params["cursor"] = f.read().strip()
            
            # Count existing articles from previous iterations
            for i in range(1, resume_from_page):
                prev_file = os.path.join(iterations_folder, f"articles_page_{i}.jsonl")
                if os.path.exists(prev_file):
                    with open(prev_file, "r") as f:
                        total_downloaded += sum(1 for _ in f)
            
            page_count = last_page
            print(f"Resuming from page {resume_from_page} with cursor: {params['cursor']}")
            
            # Open main file in append mode if resuming
            main_file_mode = "a"
        else:
            print(f"Cannot resume: cursor file for page {last_page} not found.")
            print("Starting from the beginning...")
            main_file_mode = "w"
    else:
        main_file_mode = "w"
    
    # 6. Open the main file for the entire download process
    with open(output_file, main_file_mode, encoding="utf-8") as main_file:
        has_more_pages = True
        
        while has_more_pages:
            page_count += 1
            print(f"Fetching page {page_count}...")
            
            # Make the request with current parameters
            response = requests.get(base_url, params=params)
            data = response.json()
            
            # Get the current page results
            results = data.get("results", [])
            
            # Create iteration-specific file for this batch
            iteration_file = os.path.join(iterations_folder, f"articles_page_{page_count}.jsonl")
            with open(iteration_file, "w", encoding="utf-8") as iter_file:
                # Write to both the iteration file and the main file
                for work in results:
                    json_line = json.dumps(work) + "\n"
                    iter_file.write(json_line)
                    main_file.write(json_line)
            
            total_downloaded += len(results)
            print(f"Saved {len(results)} articles to {iteration_file}")
            
            # Check if there are more pages
            cursor = data["meta"].get("next_cursor")
            if cursor:
                # Update the cursor for the next request
                params["cursor"] = cursor
                
                # Save the cursor to a file for potential resume
                cursor_file = os.path.join(iterations_folder, f"cursor_after_page_{page_count}.txt")
                with open(cursor_file, "w") as cf:
                    cf.write(cursor)
            else:
                has_more_pages = False
            
            # Optional short pause to avoid hammering the API
            time.sleep(1)
    
    print(f"\nDone! Downloaded {total_downloaded} articles in total.")
    print(f"Saved in '{output_file}'.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download articles from OpenAlex API")
    parser.add_argument("--resume", type=int, help="Resume from this page number")
    args = parser.parse_args()
    
    fetch_and_save_articles(resume_from_page=args.resume)