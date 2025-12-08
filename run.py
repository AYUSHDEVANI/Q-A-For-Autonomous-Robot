import uvicorn
import os

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Server starting...")
    print("👉 Open your browser at: http://localhost:8000")
    print("="*50 + "\n")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
