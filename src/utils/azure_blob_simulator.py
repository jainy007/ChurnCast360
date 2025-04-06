from azure.storage.blob import BlobServiceClient
import os

# Azure Azurite local connection string
AZURITE_CONNECTION_STRING = (
    "DefaultEndpointsProtocol=http;"
    "AccountName=devstoreaccount1;"
    "AccountKey=Eby8vdM02xNOcqFeqCfF3R89xWn1+IFkLd51+y6PazI=;"
    "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
)


CONTAINER_NAME = "churncast360"


def init_blob_service():
    service_client = BlobServiceClient.from_connection_string(AZURITE_CONNECTION_STRING)
    container_client = service_client.get_container_client(CONTAINER_NAME)

    # Check if container exists before creating
    try:
        container_client.get_container_properties()
        print(f"Container '{CONTAINER_NAME}' already exists.")
    except Exception:
        container_client.create_container()
        print(f"Container '{CONTAINER_NAME}' created successfully.")

    return service_client



def upload_blob(local_file_path, blob_name):
    service_client = init_blob_service()
    blob_client = service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)

    with open(local_file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
        print(f"Uploaded '{local_file_path}' as blob '{blob_name}'")


def download_blob(blob_name, download_file_path):
    service_client = init_blob_service()
    blob_client = service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)

    # Ensure directory exists
    os.makedirs(os.path.dirname(download_file_path), exist_ok=True)

    with open(download_file_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
        print(f"Downloaded blob '{blob_name}' to '{download_file_path}'")

def list_blobs():
    service_client = init_blob_service()
    container_client = service_client.get_container_client(CONTAINER_NAME)
    print("Listing blobs:")
    for blob in container_client.list_blobs():
        print(f"- {blob.name}")


if __name__ == "__main__":
    # Test uploads
    test_files = [
        ("data/raw/IBM-Telco-Customer-Churn.csv", "raw/IBM-Telco-Customer-Churn.csv"),
        ("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv", "raw/Kaggle-Telco-Customer-Churn.csv"),
        ("data/raw/bank_churn.csv", "raw/Bank-Churn.csv"),
    ]

    for local_path, blob_path in test_files:
        if os.path.exists(local_path):
            upload_blob(local_path, blob_path)
        else:
            print(f"File not found: {local_path}")

    # Test download (optional)
    download_blob("raw/IBM-Telco-Customer-Churn.csv", "downloads/IBM-Telco-Customer-Churn.csv")
    list_blobs()