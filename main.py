import argparse
from hand_detect_model import HandDetectModel
from test_model import TestModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Detection & Verification")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test model')

    args = parser.parse_args()
    if args.train: 
        handDetectModel = HandDetectModel()
        handDetectModel.training()
    testModel = TestModel()
    testModel.test_image()
 
        