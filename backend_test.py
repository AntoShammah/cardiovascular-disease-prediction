import requests
import sys
import json
from datetime import datetime

class HeartDiseaseAPITester:
    def __init__(self, base_url="https://62b6a281-2c43-4f94-9899-641625b4b9fb.preview.emergentagent.com"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name, success, details=""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - PASSED")
        else:
            print(f"❌ {name} - FAILED: {details}")
        
        self.test_results.append({
            'name': name,
            'success': success,
            'details': details
        })

    def test_api_health(self):
        """Test basic API health"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}, Response: {response.json()}" if success else f"Status: {response.status_code}"
            self.log_test("API Health Check", success, details)
            return success
        except Exception as e:
            self.log_test("API Health Check", False, str(e))
            return False

    def test_dataset_info(self):
        """Test dataset info endpoint"""
        try:
            response = requests.get(f"{self.base_url}/api/dataset-info", timeout=10)
            success = response.status_code == 200
            if success:
                data = response.json()
                required_fields = ['total_patients', 'features_count', 'target_distribution', 'dataset_shape']
                has_all_fields = all(field in data for field in required_fields)
                success = has_all_fields
                details = f"Data: {data}" if success else f"Missing fields in response: {data}"
            else:
                details = f"Status: {response.status_code}"
            
            self.log_test("Dataset Info Endpoint", success, details)
            return success, response.json() if success else {}
        except Exception as e:
            self.log_test("Dataset Info Endpoint", False, str(e))
            return False, {}

    def test_feature_importance(self):
        """Test feature importance endpoint"""
        try:
            response = requests.get(f"{self.base_url}/api/feature-importance", timeout=15)
            success = response.status_code == 200
            if success:
                data = response.json()
                has_feature_importance = 'feature_importance' in data
                if has_feature_importance:
                    features = data['feature_importance']
                    is_list = isinstance(features, list)
                    has_features = len(features) > 0 if is_list else False
                    if has_features and is_list:
                        first_feature = features[0]
                        has_required_fields = 'feature' in first_feature and 'score' in first_feature
                        success = has_required_fields
                        details = f"Found {len(features)} features" if success else "Missing required fields in feature data"
                    else:
                        success = False
                        details = "No features found or invalid format"
                else:
                    success = False
                    details = "Missing 'feature_importance' in response"
            else:
                details = f"Status: {response.status_code}"
            
            self.log_test("Feature Importance Endpoint", success, details)
            return success
        except Exception as e:
            self.log_test("Feature Importance Endpoint", False, str(e))
            return False

    def test_model_training(self):
        """Test model training endpoint"""
        try:
            print("🔄 Training models (this may take a while)...")
            response = requests.post(f"{self.base_url}/api/train-models", timeout=60)
            success = response.status_code == 200
            if success:
                data = response.json()
                has_results = 'model_results' in data and 'best_model' in data
                if has_results:
                    results = data['model_results']
                    is_list = isinstance(results, list)
                    has_models = len(results) > 0 if is_list else False
                    if has_models and is_list:
                        first_model = results[0]
                        required_fields = ['algorithm', 'accuracy', 'f1_score', 'precision', 'recall']
                        has_required_fields = all(field in first_model for field in required_fields)
                        success = has_required_fields
                        details = f"Trained {len(results)} models, Best: {data['best_model']}" if success else "Missing required fields in model results"
                    else:
                        success = False
                        details = "No model results found"
                else:
                    success = False
                    details = "Missing 'model_results' or 'best_model' in response"
            else:
                details = f"Status: {response.status_code}"
            
            self.log_test("Model Training Endpoint", success, details)
            return success, data if success else {}
        except Exception as e:
            self.log_test("Model Training Endpoint", False, str(e))
            return False, {}

    def test_model_comparison(self):
        """Test model comparison endpoint"""
        try:
            response = requests.get(f"{self.base_url}/api/model-comparison", timeout=30)
            success = response.status_code == 200
            if success:
                data = response.json()
                has_results = 'model_results' in data and 'best_model' in data
                success = has_results
                details = f"Best model: {data.get('best_model', 'Unknown')}" if success else "Missing required fields"
            else:
                details = f"Status: {response.status_code}"
            
            self.log_test("Model Comparison Endpoint", success, details)
            return success
        except Exception as e:
            self.log_test("Model Comparison Endpoint", False, str(e))
            return False

    def test_heart_disease_prediction(self):
        """Test heart disease prediction with sample patient data"""
        # Sample patient data as specified in the review request
        sample_patient = {
            "age": 55,
            "gender": 1,  # Male
            "cigsPerDay": 10,
            "sysBP": 140,
            "diaBP": 90,
            "totChol": 240,
            "prevalentHyp": 1,  # Yes
            "diabetes": 0,  # No
            "glucose": 95,
            "BPMeds": 1  # Yes
        }
        
        try:
            response = requests.post(f"{self.base_url}/api/predict", json=sample_patient, timeout=15)
            success = response.status_code == 200
            if success:
                data = response.json()
                required_fields = ['prediction', 'risk_probability', 'model_used', 'interpretation']
                has_required_fields = all(field in data for field in required_fields)
                if has_required_fields:
                    prediction_valid = data['prediction'] in [0, 1]
                    probability_valid = 0 <= data['risk_probability'] <= 100
                    interpretation_valid = data['interpretation'] in ['High Risk', 'Low Risk']
                    success = prediction_valid and probability_valid and interpretation_valid
                    details = f"Prediction: {data['interpretation']}, Probability: {data['risk_probability']}%, Model: {data['model_used']}" if success else "Invalid prediction values"
                else:
                    success = False
                    details = f"Missing required fields. Got: {list(data.keys())}"
            else:
                details = f"Status: {response.status_code}"
            
            self.log_test("Heart Disease Prediction", success, details)
            return success, data if success else {}
        except Exception as e:
            self.log_test("Heart Disease Prediction", False, str(e))
            return False, {}

    def test_prediction_with_invalid_data(self):
        """Test prediction with invalid data to check error handling"""
        invalid_patient = {
            "age": "invalid",  # Should be number
            "gender": 1,
            "cigsPerDay": 10
            # Missing required fields
        }
        
        try:
            response = requests.post(f"{self.base_url}/api/predict", json=invalid_patient, timeout=10)
            # Should return 422 (validation error) or 500 (server error)
            success = response.status_code in [422, 500]
            details = f"Status: {response.status_code} (Expected error response)"
            self.log_test("Invalid Data Error Handling", success, details)
            return success
        except Exception as e:
            self.log_test("Invalid Data Error Handling", False, str(e))
            return False

    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting Heart Disease API Tests")
        print("=" * 50)
        
        # Test 1: Basic API health
        if not self.test_api_health():
            print("❌ API is not responding. Stopping tests.")
            return False
        
        # Test 2: Dataset info
        dataset_success, dataset_info = self.test_dataset_info()
        if dataset_success:
            print(f"📊 Dataset: {dataset_info.get('total_patients', 'Unknown')} patients, {dataset_info.get('features_count', 'Unknown')} features")
        
        # Test 3: Feature importance
        self.test_feature_importance()
        
        # Test 4: Model training (this trains the models needed for prediction)
        training_success, training_data = self.test_model_training()
        
        # Test 5: Model comparison
        self.test_model_comparison()
        
        # Test 6: Heart disease prediction (only if models are trained)
        if training_success:
            prediction_success, prediction_data = self.test_heart_disease_prediction()
            if prediction_success:
                print(f"🔮 Sample Prediction: {prediction_data.get('interpretation', 'Unknown')} ({prediction_data.get('risk_probability', 0)}% risk)")
        else:
            print("⚠️  Skipping prediction test due to model training failure")
        
        # Test 7: Error handling
        self.test_prediction_with_invalid_data()
        
        # Print summary
        print("\n" + "=" * 50)
        print("📋 TEST SUMMARY")
        print("=" * 50)
        print(f"Total Tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        # Print failed tests
        failed_tests = [test for test in self.test_results if not test['success']]
        if failed_tests:
            print("\n❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"  - {test['name']}: {test['details']}")
        
        return self.tests_passed == self.tests_run

def main():
    """Main test function"""
    tester = HeartDiseaseAPITester()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())