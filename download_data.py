import os
import shutil
from datasets import load_dataset, Pdf
from tqdm import tqdm

def download_dataset_pdfs(dataset_name, output_dir):
    print(f"\nLoading dataset '{dataset_name}'...")
    dataset = load_dataset(dataset_name)
    
    # Avoid decoding the PDFs into text and keep them as raw bytes
    dataset = dataset.cast_column("pdf", Pdf(decode=False))
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Extracting PDFs to '{output_dir}' directory...")
    
    # Iterate over the train split and save each PDF
    for i, item in enumerate(tqdm(dataset["train"], desc=f"Saving to {output_dir}")):
        pdf_data = item["pdf"]
        
        # Get the original file name if available, otherwise create one
        filename = pdf_data.get("path")
        if not filename:
            filename = f"document_{i}.pdf"
        
        # Clean up the filename to just be the basename
        filename = os.path.basename(filename)
        filepath = os.path.join(output_dir, filename)
        
        if pdf_data.get("bytes") is not None:
            with open(filepath, "wb") as f:
                f.write(pdf_data["bytes"])
        elif pdf_data.get("path") is not None and os.path.exists(pdf_data["path"]):
            shutil.copy2(pdf_data["path"], filepath)
        else:
            print(f"Warning: Could not find bytes or path for {filename}")
            
    print(f"Successfully saved {len(dataset['train'])} PDFs to '{os.path.abspath(output_dir)}'.")

if __name__ == "__main__":
    # Download invoices
    download_dataset_pdfs("AyoubChLin/northwind_invocies", "invoices")

    # Download shipping orders
    download_dataset_pdfs("AyoubChLin/northwind_Shipping_orders", "shipping_orders")
