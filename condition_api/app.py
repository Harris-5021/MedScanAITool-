from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from gradcam_utils import generate_gradcam_map
import pyodbc
from datetime import datetime
from transformers import ViTModel

app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ensure the static directory exists
if not os.path.exists("static/pdfs"):
    os.makedirs("static/pdfs")

# Database connection
conn_str = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=medscan;"
    "Trusted_Connection=yes;"
    "Encrypt=yes;"
    "TrustServerCertificate=yes;"
)

try:
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    print("Database connection successful!")
except pyodbc.Error as e:
    raise RuntimeError(f"Failed to connect to the database: {e}")

# Define the PretrainedVisionTransformer class
class PretrainedVisionTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.head = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.head(cls_output)

# Load your model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model = PretrainedVisionTransformer(num_classes=5)
    state_dict = torch.load("models/multilabel_vit_model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Preprocessing transform
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Denormalize transform
denormalize = transforms.Compose([
    transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
    transforms.ToPILImage(),
])

# Conditions list
conditions = ["normal", "pneumonia_bacterial", "pneumonia_viral", "pneumothorax", "tuberculosis"]

def get_region_description(attention_map):
    try:
        attention_map = np.array(attention_map)
        h, w = attention_map.shape
        top_half = attention_map[:h//2, :]
        bottom_half = attention_map[h//2:, :]
        left_half = attention_map[:, :w//2]
        right_half = attention_map[:, w//2:]

        # Calculate the dominant quadrant in image terms
        quadrants = {
            "upper left": top_half[:, :w//2].sum(),
            "upper right": top_half[:, w//2:].sum(),
            "lower left": bottom_half[:, :w//2].sum(),
            "lower right": bottom_half[:, w//2:].sum(),
        }

        dominant_region = max(quadrants, key=quadrants.get)

        # Map image-based quadrant to patient's anatomical perspective
        anatomical_mapping = {
            "upper left": "patient's upper right lung",
            "upper right": "patient's upper left lung",
            "lower left": "patient's lower right lung",
            "lower right": "patient's lower left lung",
        }

        return anatomical_mapping[dominant_region]
    except Exception as e:
        return f"Error determining region: {e}"

@app.post("/predict")
async def predict(
    user_id: int = Form(...),
    file: UploadFile = File(...),
    patient_name: str = Form(...),
    patient_age: int = Form(None),
    patient_gender: str = Form(None)
):
    # Verify user exists
    try:
        cursor.execute("SELECT id FROM dbo.users WHERE id = ?", user_id)
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="User not found")
    except pyodbc.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    # Insert patient details into dbo.patients
    try:
        cursor.execute(
            "INSERT INTO dbo.patients (user_id, name, age, gender) VALUES (?, ?, ?, ?)",
            (user_id, patient_name, patient_age, patient_gender)
        )
        conn.commit()
        cursor.execute("SELECT @@IDENTITY AS id")
        patient_id = int(cursor.fetchone()[0])
    except pyodbc.Error as e:
        raise HTTPException(status_code=500, detail=f"Failed to save patient details: {e}")

    # Create user-specific directory
    user_pdf_dir = f"static/pdfs/user_{user_id}"
    try:
        if not os.path.exists(user_pdf_dir):
            os.makedirs(user_pdf_dir)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to create user directory: {e}")

    # Load and preprocess the image
    try:
        image = Image.open(file.file).convert("RGB")
        image_tensor = preprocess(image).to(device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {e}")

    # Save the original image
    original_image = denormalize(image_tensor.cpu())
    original_image_path = f"{user_pdf_dir}/original_xray.png"
    try:
        original_image.save(original_image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save original image: {e}")

    # Ensure batch dimension
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)

    # Get model predictions
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            predictions = torch.sigmoid(outputs).cpu().numpy()[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Generate Grad-CAM map for the top predicted condition
    top_condition_idx = predictions.argmax()
    top_condition = conditions[top_condition_idx]
    try:
        gradcam_image, attention_map = generate_gradcam_map(model, image_tensor, class_idx=top_condition_idx)
        gradcam_path = f"{user_pdf_dir}/gradcam_{top_condition}.png"
        gradcam_image.save(gradcam_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate Grad-CAM: {e}")

    # Get the dominant region
    dominant_region = get_region_description(attention_map)

    # Generate a PDF report with a unique name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"user_{user_id}_{timestamp}.pdf"
    pdf_path = f"{user_pdf_dir}/{pdf_filename}"
    report_date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")  # Format report date in UK format
    try:
        c = canvas.Canvas(pdf_path, pagesize=letter)
        
        # Add the MedScan logo at the top-left
        logo_path = "static/images/medscan_logo.jpg"
        if not os.path.exists(logo_path):
            raise FileNotFoundError(f"Logo file not found at {logo_path}")
        c.drawImage(logo_path, 50, 750, width=100, height=40)  # Adjust position and size as needed
        
        # Add a header with branding (below the logo)
        c.setFont("Helvetica-Bold", 20)
        c.setFillColorRGB(0.2, 0.2, 0.6)  # Dark blue color
        c.drawString(50, 710, "MedScan X-Ray Analysis Report")
        
        # Draw a horizontal line under the header
        c.setLineWidth(1)
        c.setStrokeColorRGB(0.2, 0.2, 0.6)
        c.line(50, 700, 550, 700)
        
        # Add metadata
        c.setFont("Helvetica", 10)
        c.setFillColorRGB(0, 0, 0)  # Black color
        y = 670
        c.drawString(50, y, f"User ID: {user_id}")
        y -= 15
        c.drawString(50, y, f"Report Date: {report_date}")
        y -= 15
        
        # Add patient details
        c.setFont("Helvetica-Bold", 12)
        y -= 15
        c.drawString(50, y, "Patient Details:")
        y -= 15
        c.setFont("Helvetica", 10)
        c.drawString(50, y, f"Name: {patient_name}")
        y -= 15
        if patient_age is not None:
            c.drawString(50, y, f"Age: {patient_age}")
            y -= 15
        if patient_gender:
            c.drawString(50, y, f"Gender: {patient_gender}")
            y -= 15
        
        # Add prediction results as percentages
        c.setFont("Helvetica-Bold", 12)
        y -= 20
        c.drawString(50, y, "Prediction Probabilities:")
        y -= 20
        for idx, cond in enumerate(conditions):
            prob = predictions[idx] * 100
            if idx == top_condition_idx:  # Highlight the top prediction
                c.setFillColorRGB(1, 0, 0)  # Red color
                c.setFont("Helvetica-Bold", 12)
            else:
                c.setFillColorRGB(0, 0, 0)  # Black color
                c.setFont("Helvetica", 12)
            c.drawString(50, y, f"{cond}: {prob:.1f}%")
            y -= 20
        
        # Reset color for remaining text
        c.setFillColorRGB(0, 0, 0)
        
        # Check if there's enough space for the images (200 height + 20 for labels + extra space)
        if y - 240 < 100:  # Leave 100 units for footer, 240 = 200 (image height) + 20 (labels) + 20 (extra space)
            c.showPage()
            y = 750  # Reset y to top of new page
        
        # Add images side by side
        c.setFont("Helvetica", 12)
        y -= 20
        c.drawString(50, y, "Original X-Ray:")
        c.drawString(300, y, f"Grad-CAM ({top_condition}):")
        y -= 220  # Space for images (200 height) + 20 extra space for gap
        c.drawImage(original_image_path, 50, y, width=200, height=200)
        c.drawImage(gradcam_path, 300, y, width=200, height=200)

        # Check if there's enough space for the remaining sections
        if y - 100 < 100:  # Leave 100 units for footer
            c.showPage()
            y = 750  # Reset y to top of new page
        
        # Add region description
        c.setFont("Helvetica-Bold", 12)
        y -= 30
        c.drawString(50, y, f"Region of Interest: {dominant_region}")

        # Add summary
        c.setFont("Helvetica", 12)
        y -= 30
        c.drawString(50, y, "Summary:")
        y -= 20
        c.drawString(50, y, f"The model predicts a {predictions[top_condition_idx]*100:.1f}% likelihood of {top_condition},")
        y -= 15
        c.drawString(50, y, f"with the region of interest in the {dominant_region}.")

        # Add disclaimer
        c.setFont("Helvetica-Oblique", 10)
        c.setFillColorRGB(0.5, 0.5, 0.5)  # Gray color
        y -= 30
        c.drawString(50, y, "Disclaimer: This is an AI-generated report and should be reviewed by a medical professional.")

        # Add footer on the last page
        c.setFont("Helvetica", 8)
        c.drawString(50, 50, "For support, contact support@medscan.com or visit www.medscan.com")

        c.save()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {e}")

    # Store the PDF path in the database with patient_id
    try:
        cursor.execute(
            "INSERT INTO dbo.pdf (user_id, patient_id, file_path, created_at) VALUES (?, ?, ?, ?)",
            (user_id, patient_id, pdf_path, datetime.now())
        )
        conn.commit()
        cursor.execute("SELECT @@IDENTITY AS id")
        pdf_id = int(cursor.fetchone()[0])
    except pyodbc.Error as e:
        raise HTTPException(status_code=500, detail=f"Failed to save PDF path to database: {e}")

    # Re-open the PDF to add the pdf_id (Report ID)
    try:
        c = canvas.Canvas(pdf_path, pagesize=letter)
        
        # Add the MedScan logo at the top-left
        c.drawImage(logo_path, 50, 750, width=100, height=40)  # Adjust position and size as needed
        
        # Add a header with branding (below the logo)
        c.setFont("Helvetica-Bold", 20)
        c.setFillColorRGB(0.2, 0.2, 0.6)  # Dark blue color
        c.drawString(50, 710, "MedScan X-Ray Analysis Report")
        
        # Draw a horizontal line under the header
        c.setLineWidth(1)
        c.setStrokeColorRGB(0.2, 0.2, 0.6)
        c.line(50, 700, 550, 700)
        
        # Add metadata with pdf_id
        c.setFont("Helvetica", 10)
        c.setFillColorRGB(0, 0, 0)  # Black color
        y = 670
        c.drawString(50, y, f"User ID: {user_id}")
        y -= 15
        c.drawString(50, y, f"Report Date: {report_date}")
        y -= 15
        c.drawString(50, y, f"Report ID: {pdf_id}")
        y -= 15
        
        # Add patient details
        c.setFont("Helvetica-Bold", 12)
        y -= 15
        c.drawString(50, y, "Patient Details:")
        y -= 15
        c.setFont("Helvetica", 10)
        c.drawString(50, y, f"Name: {patient_name}")
        y -= 15
        if patient_age is not None:
            c.drawString(50, y, f"Age: {patient_age}")
            y -= 15
        if patient_gender:
            c.drawString(50, y, f"Gender: {patient_gender}")
            y -= 15
        
        # Add prediction results as percentages
        c.setFont("Helvetica-Bold", 12)
        y -= 20
        c.drawString(50, y, "Prediction Probabilities:")
        y -= 20
        for idx, cond in enumerate(conditions):
            prob = predictions[idx] * 100
            if idx == top_condition_idx:  # Highlight the top prediction
                c.setFillColorRGB(1, 0, 0)  # Red color
                c.setFont("Helvetica-Bold", 12)
            else:
                c.setFillColorRGB(0, 0, 0)  # Black color
                c.setFont("Helvetica", 12)
            c.drawString(50, y, f"{cond}: {prob:.1f}%")
            y -= 20
        
        # Reset color for remaining text
        c.setFillColorRGB(0, 0, 0)
        
        # Check if there's enough space for the images
        if y - 240 < 100:
            c.showPage()
            y = 750
        
        # Add images side by side
        c.setFont("Helvetica", 12)
        y -= 20
        c.drawString(50, y, "Original X-Ray:")
        c.drawString(300, y, f"Grad-CAM ({top_condition}):")
        y -= 220  # Space for images (200 height) + 20 extra space for gap
        c.drawImage(original_image_path, 50, y, width=200, height=200)
        c.drawImage(gradcam_path, 300, y, width=200, height=200)

        # Check if there's enough space for the remaining sections
        if y - 100 < 100:
            c.showPage()
            y = 750
        
        # Add region description
        c.setFont("Helvetica-Bold", 12)
        y -= 30
        c.drawString(50, y, f"Region of Interest: {dominant_region}")

        # Add summary
        c.setFont("Helvetica", 12)
        y -= 30
        c.drawString(50, y, "Summary:")
        y -= 20
        c.drawString(50, y, f"The model predicts a {predictions[top_condition_idx]*100:.1f}% likelihood of {top_condition},")
        y -= 15
        c.drawString(50, y, f"with the region of interest in the {dominant_region}.")

        # Add disclaimer
        c.setFont("Helvetica-Oblique", 10)
        c.setFillColorRGB(0.5, 0.5, 0.5)  # Gray color
        y -= 30
        c.drawString(50, y, "Disclaimer: This is an AI-generated report and should be reviewed by a medical professional.")

        # Add footer on the last page
        c.setFont("Helvetica", 8)
        c.drawString(50, 50, "For support, contact support@medscan.com or visit www.medscan.com")

        c.save()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update PDF with Report ID: {e}")

    # Return the PDF URL
    return JSONResponse(content={"pdf_url": pdf_path})

@app.get("/user_pdfs/{user_id}")
async def get_user_pdfs(user_id: int):
    # Verify user exists
    try:
        cursor.execute("SELECT id FROM dbo.users WHERE id = ?", user_id)
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="User not found")
    except pyodbc.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    # Retrieve PDFs for the user with patient details
    try:
        cursor.execute("""
            SELECT p.pdf_id, p.file_path, p.created_at, pt.name, pt.age, pt.gender
            FROM dbo.pdf p
            LEFT JOIN dbo.patients pt ON p.patient_id = pt.patient_id
            WHERE p.user_id = ?
            ORDER BY p.created_at DESC
        """, user_id)
        pdfs = [
            {
                "pdf_id": row[0],
                "file_path": row[1],
                "created_at": row[2].isoformat(),
                "patient_name": row[3],
                "patient_age": row[4],
                "patient_gender": row[5]
            }
            for row in cursor.fetchall()
        ]
        return JSONResponse(content={"pdfs": pdfs})
    except pyodbc.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

# Ensure database connection is closed on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    cursor.close()
    conn.close()