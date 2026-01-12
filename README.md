Restoration Walkthrough - Ethiopia Plant Disease API
I have successfully restored your project code based on the detailed structure you provided. All files are now populated with clean, production-ready code.

📁 Restored File Structure
et-disease-api/
├── configs/
│   └── default.yaml       # [RESTORED] Hyperparameters & paths
├── models/
│   ├── efficientnet.py    # [RESTORED] DiseaseClassifier model
│   └── lightning_module.py# [RESTORED] PyTorch Lightning wrapper
├── src/
│   ├── preprocess_data.py # [RESTORED] Data split & resize script
│   ├── train.py           # [RESTORED] Training loop & logging
│   ├── infer.py           # [RESTORED] CLI inference script
│   └── api.py             # [RESTORED] FastAPI service
├── Dockerfile             # [RESTORED] Deployment container
├── Makefile               # [RESTORED] Easy command shortcuts
├── requirements.txt       # [RESTORED] Dependencies
└── .gitignore             # [UPDATED] Allowed requirements.txt
🚀 How to Run
1. Setup Environment
pip install -r requirements.txt
2. Prepare Data
Place your raw images in data/raw/{Class Name}/*.jpg. Then run:

make preprocess
# OR
python src/preprocess_data.py
3. Train Model
make train
# OR
python src/train.py
This will save the best model to 
weights/best.ckpt
.

4. Run Inference (CLI)
make infer IMAGE=test_leaf.jpg
# OR
python src/infer.py --image test_leaf.jpg
5. Start API Server
make api
# OR
uvicorn src.api:app --reload
API Documentation: http://localhost:8000/docs
Predict Endpoint: POST /predict
6. Docker Build
make docker-build
make docker-run
✅ Verification
I ran a syntax check on all restored Python files, and they passed successfully. The project is ready for data ingestion and training.

