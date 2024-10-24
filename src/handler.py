from dotenv import load_dotenv
import boto3
import pandas as pd
import os
from pathlib import Path
import numpy as np
from botocore.exceptions import NoCredentialsError
import cv2
import runpod
from runpod.serverless.utils import rp_download
from pdf2image import convert_from_path
from main import table_extractor

load_dotenv()

# Initialize S3 client
s3_client = boto3.client('s3', region_name='ap-south-1')
s3_bucket_name = os.getenv('S3_BUCKET_NAME')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
print(s3_bucket_name)


def upload_file_to_s3(file_path: Path):
    """Uploads a file to S3 and returns the presigned URL."""
    s3_client = boto3.client('s3', region_name='ap-south-1')
    file_key = f"bankstmt-ocr-demo/{file_path}"  # Customize this to match your S3 folder structure
    try:
        s3_client.upload_file(str(file_path), s3_bucket_name, file_key)
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': s3_bucket_name, 'Key': file_key},
            ExpiresIn=3600  # URL expiration time in seconds
        )
        return presigned_url
    except FileNotFoundError:

        return None
    except NoCredentialsError:
        return None


def handler(job):
    job_input = job.get('input', {})

    if not job_input.get("file_path", False):
        return {
            "error": "Input is missing the 'file_path' key. Please include a file_path and retry your request."
        }

    file_path = job_input.get("file_path")
    file_name = os.path.basename(file_path)
    ocr = job_input.get("ocr")
    print(ocr)

    #downloaded_file = rp_download.file(file_path)
    #file_path = downloaded_file.get('file_path')

    filename = os.path.basename(file_path).split('.')[0]
    if file_path.lower().endswith('.pdf'):
        images = convert_from_path(file_path)
        pages = len(images)
    else:
        images = [cv2.imread(file_path)]
        pages = 1

    output_excel_path = f"{filename}_{ocr}.xlsx"
    writer = pd.ExcelWriter(output_excel_path, engine='xlsxwriter')
    html = ""  # Initialize html here

    for page_number in range(pages):
        image = images[page_number]
        if not isinstance(image, np.ndarray):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        detection_result = table_extractor.table_detection(image)
        crop_images = table_extractor.crop_image(image, detection_result)

        for table_number, crop_image in enumerate(crop_images):
            words = []
            if ocr == 'doctr':
                words = table_extractor.doctr(crop_image)
            elif ocr == 'paddle':
                words = table_extractor.ocr(crop_image)

            structure_result = table_extractor.table_structure(crop_image)
            table_structures, cells, confidence_score = table_extractor.convert_structure(words, crop_image,
                                                                                          structure_result)
            data_rows = table_extractor.visualize_cells(crop_image, table_structures, cells)
            sheet_name = f'Page_{page_number + 1}_Table_{table_number + 1}'
            df = pd.DataFrame(data_rows)
            df = df.rename(columns=df.iloc[0]).drop(df.index[0])
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            html = table_extractor.cells_to_html(cells)

    writer.close()

    download_url = upload_file_to_s3(output_excel_path)

    # Clean up the local file if necessary
    if os.path.exists(output_excel_path):
        os.remove(output_excel_path)

    return {"refresh_worker": False, "job_results": download_url}


# Run the handler locally for testing
if __name__ == '__main__':
    job = {
        "input": {
            "file_path": "wells_fargo_USA_page-0002.jpg",
            "ocr": "doctr"
        }
    }
    result = handler(job)
    print(result)
