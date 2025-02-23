import os, sys
from pathlib import Path
from werkzeug.utils import secure_filename
from src.exception.operationhandler import system_logger


allowed_files = ["txt", "csv", "json", "pdf", "doc", "docx", "pptx"]

def _allowed_file(filename:str):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_files

def _file_checks(files:str):
    if not files:
        return {
            "detail" : "No file found",
            "status_code" : 400
        }
    for file in files:
        if file is None or getattr(file, "filenames", "") == "":
            return {
            "detail" : "No file found",
            "status_code" : 400
        }
        if not _allowed_file(file.filename):
            return {
                "detail": f"File format not supported. use any of {allowed_files}",
                "status_code": 415
            }
    
    return {
        "detail" : "success",
        "status_code": 200
    }

async def _upload_files(files:str, temp_dir:Path):
    checks = _file_checks(files)
    if checks["status_code"] != 200:
        return checks
    try:
        for file in files:
            filename = secure_filename(file.filenames)
            file_path = os.path.join(temp_dir,filename)

            file_obj = await file.read()

            with open(file_path, "wb") as buffer:
                buffer.write(file_obj)
            
        return {
                "detail" : "Upload completed",
                "status_code": 200
            }
        
    except FileNotFoundError:
        system_logger.exception("File not found during upload.")
        return {"detail": "File handling error", "status_code": 500}
    except Exception as e:
        system_logger.exception(f"Unexpected error during upload: {e}")
        return {"detail": "Internal server error", "status_code": 500}



