import os
from supabase import create_client, Client

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

supabase: Client = None

if url and key:
    # Debug logging to investigate Render "Invalid API key" error
    print(f"DEBUG: SUPABASE_KEY length: {len(key)}")
    print(f"DEBUG: SUPABASE_KEY first 5 chars: '{key[:5]}...'")
    has_quotes = '"' in key or "'" in key
    print(f"DEBUG: Quotes present in key? {'Yes' if has_quotes else 'No'}")
    
    try:
        supabase = create_client(url, key)
    except Exception as e:
        print(f"DEBUG: Error creating Supabase client: {e}")
        raise e
else:
    print("Warning: SUPABASE_URL or SUPABASE_KEY not found. Supabase features will be disabled.")
