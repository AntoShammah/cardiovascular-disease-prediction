import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ArcElement } from 'chart.js';
import { Bar, Doughnut } from 'react-chartjs-2';
import { Heart, Activity, BarChart3, Brain, Stethoscope, AlertTriangle, CheckCircle, Loader } from 'lucide-react';
import toast, { Toaster } from 'react-hot-toast';
import './App.css';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ArcElement);

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

function App() {
  const [activeTab, setActiveTab] = useState('prediction');
  const [loading, setLoading] = useState(false);
  const [modelResults, setModelResults] = useState([]);
  const [featureImportance, setFeatureImportance] = useState([]);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [bestModel, setBestModel] = useState('');
  
  // Patient form data
  const [patientData, setPatientData] = useState({
    age: '',
    gender: '',
    cigsPerDay: '',
    sysBP: '',
    diaBP: '',
    totChol: '',
    prevalentHyp: '',
    diabetes: '',
    glucose: '',
    BPMeds: ''
  });
  
  // Prediction result
  const [predictionResult, setPredictionResult] = useState(null);

  // Load initial data
  useEffect(() => {
    loadDatasetInfo();
  }, []);

  const loadDatasetInfo = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/dataset-info`);
      setDatasetInfo(response.data);
    } catch (error) {
      console.error('Error loading dataset info:', error);
      toast.error('Failed to load dataset information');
    }
  };

  const loadFeatureImportance = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${BACKEND_URL}/api/feature-importance`);
      setFeatureImportance(response.data.feature_importance);
      toast.success('Feature importance analysis completed!');
    } catch (error) {
      console.error('Error loading feature importance:', error);
      toast.error('Failed to load feature importance');
    }
    setLoading(false);
  };

  const trainModels = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${BACKEND_URL}/api/train-models`);
      setModelResults(response.data.model_results);
      setBestModel(response.data.best_model);
      toast.success('All models trained successfully!');
    } catch (error) {
      console.error('Error training models:', error);
      toast.error('Failed to train models');
    }
    setLoading(false);
  };

  const getModelComparison = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${BACKEND_URL}/api/model-comparison`);
      setModelResults(response.data.model_results);
      setBestModel(response.data.best_model);
      toast.success('Model comparison loaded!');
    } catch (error) {
      console.error('Error loading model comparison:', error);
      toast.error('Failed to load model comparison');
    }
    setLoading(false);
  };

  const predictHeartDisease = async () => {
    // Validate form
    const requiredFields = Object.keys(patientData);
    const emptyFields = requiredFields.filter(field => !patientData[field]);
    
    if (emptyFields.length > 0) {
      toast.error('Please fill in all patient information fields');
      return;
    }

    setLoading(true);
    try {
      const numericData = Object.fromEntries(
        Object.entries(patientData).map(([key, value]) => [key, parseFloat(value)])
      );
      
      const response = await axios.post(`${BACKEND_URL}/api/predict`, numericData);
      setPredictionResult(response.data);
      toast.success('Prediction completed!');
    } catch (error) {
      console.error('Error making prediction:', error);
      toast.error('Failed to make prediction');
    }
    setLoading(false);
  };

  const handleInputChange = (field, value) => {
    setPatientData(prev => ({ ...prev, [field]: value }));
  };

  const resetForm = () => {
    setPatientData({
      age: '',
      gender: '',
      cigsPerDay: '',
      sysBP: '',
      diaBP: '',
      totChol: '',
      prevalentHyp: '',
      diabetes: '',
      glucose: '',
      BPMeds: ''
    });
    setPredictionResult(null);
  };

  // Chart configurations
  const modelComparisonChart = {
    labels: modelResults.map(result => result.algorithm),
    datasets: [
      {
        label: 'Accuracy (%)',
        data: modelResults.map(result => result.accuracy),
        backgroundColor: [
          'rgba(14, 165, 233, 0.8)',
          'rgba(34, 197, 94, 0.8)',
          'rgba(245, 158, 11, 0.8)',
          'rgba(239, 68, 68, 0.8)',
          'rgba(168, 85, 247, 0.8)',
          'rgba(6, 182, 212, 0.8)',
          'rgba(217, 119, 6, 0.8)',
          'rgba(220, 38, 38, 0.8)'
        ],
        borderColor: [
          'rgb(14, 165, 233)',
          'rgb(34, 197, 94)',
          'rgb(245, 158, 11)',
          'rgb(239, 68, 68)',
          'rgb(168, 85, 247)',
          'rgb(6, 182, 212)',
          'rgb(217, 119, 6)',
          'rgb(220, 38, 38)'
        ],
        borderWidth: 2,
        borderRadius: 8,
      }
    ]
  };

  const featureChart = {
    labels: featureImportance.slice(0, 10).map(item => item.feature),
    datasets: [
      {
        label: 'Feature Importance Score',
        data: featureImportance.slice(0, 10).map(item => item.score),
        backgroundColor: 'rgba(34, 197, 94, 0.8)',
        borderColor: 'rgb(34, 197, 94)',
        borderWidth: 2,
        borderRadius: 8,
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Model Performance Comparison',
        font: {
          size: 16,
          weight: 'bold'
        }
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          callback: function(value) {
            return value + '%';
          }
        }
      }
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <Toaster position="top-right" />
      
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-medical-100 rounded-xl">
                <Heart className="h-8 w-8 text-medical-600" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-slate-800">CardioPredict</h1>
                <p className="text-sm text-slate-600">AI-Powered Heart Disease Prediction System</p>
              </div>
            </div>
            
            {datasetInfo && (
              <div className="flex items-center space-x-4 text-sm text-slate-600">
                <div className="flex items-center space-x-1">
                  <Activity className="h-4 w-4" />
                  <span>{datasetInfo.total_patients} Patients</span>
                </div>
                <div className="flex items-center space-x-1">
                  <BarChart3 className="h-4 w-4" />
                  <span>{datasetInfo.features_count} Features</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {[
              { id: 'prediction', label: 'Patient Prediction', icon: Stethoscope },
              { id: 'features', label: 'Feature Analysis', icon: Brain },
              { id: 'models', label: 'Model Comparison', icon: BarChart3 },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 py-4 px-2 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab.id
                    ? 'border-medical-500 text-medical-600'
                    : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'
                }`}
              >
                <tab.icon className="h-4 w-4" />
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* Patient Prediction Tab */}
        {activeTab === 'prediction' && (
          <div className="space-y-8">
            <div className="text-center">
              <h2 className="section-title">Heart Disease Risk Assessment</h2>
              <p className="text-slate-600 max-w-2xl mx-auto">
                Enter patient information below to get an AI-powered cardiovascular risk assessment using advanced machine learning models.
              </p>
            </div>

            <div className="grid lg:grid-cols-2 gap-8">
              {/* Input Form */}
              <div className="form-section">
                <h3 className="subsection-title flex items-center space-x-2">
                  <Stethoscope className="h-5 w-5 icon-medical" />
                  <span>Patient Information</span>
                </h3>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="input-group">
                    <label className="input-label">Age (years)</label>
                    <input
                      type="number"
                      value={patientData.age}
                      onChange={(e) => handleInputChange('age', e.target.value)}
                      placeholder="e.g., 45"
                      className="input-field"
                      min="1"
                      max="120"
                    />
                  </div>
                  
                  <div className="input-group">
                    <label className="input-label">Gender</label>
                    <select
                      value={patientData.gender}
                      onChange={(e) => handleInputChange('gender', e.target.value)}
                      className="input-field"
                    >
                      <option value="">Select Gender</option>
                      <option value="1">Male</option>
                      <option value="0">Female</option>
                    </select>
                  </div>
                  
                  <div className="input-group">
                    <label className="input-label">Cigarettes per Day</label>
                    <input
                      type="number"
                      value={patientData.cigsPerDay}
                      onChange={(e) => handleInputChange('cigsPerDay', e.target.value)}
                      placeholder="e.g., 0"
                      className="input-field"
                      min="0"
                      max="100"
                    />
                  </div>
                  
                  <div className="input-group">
                    <label className="input-label">Systolic BP (mmHg)</label>
                    <input
                      type="number"
                      value={patientData.sysBP}
                      onChange={(e) => handleInputChange('sysBP', e.target.value)}
                      placeholder="e.g., 120"
                      className="input-field"
                      min="70"
                      max="300"
                    />
                  </div>
                  
                  <div className="input-group">
                    <label className="input-label">Diastolic BP (mmHg)</label>
                    <input
                      type="number"
                      value={patientData.diaBP}
                      onChange={(e) => handleInputChange('diaBP', e.target.value)}
                      placeholder="e.g., 80"
                      className="input-field"
                      min="40"
                      max="200"
                    />
                  </div>
                  
                  <div className="input-group">
                    <label className="input-label">Cholesterol (mg/dL)</label>
                    <input
                      type="number"
                      value={patientData.totChol}
                      onChange={(e) => handleInputChange('totChol', e.target.value)}
                      placeholder="e.g., 200"
                      className="input-field"
                      min="100"
                      max="600"
                    />
                  </div>
                  
                  <div className="input-group">
                    <label className="input-label">Hypertensive</label>
                    <select
                      value={patientData.prevalentHyp}
                      onChange={(e) => handleInputChange('prevalentHyp', e.target.value)}
                      className="input-field"
                    >
                      <option value="">Select</option>
                      <option value="1">Yes</option>
                      <option value="0">No</option>
                    </select>
                  </div>
                  
                  <div className="input-group">
                    <label className="input-label">Diabetes</label>
                    <select
                      value={patientData.diabetes}
                      onChange={(e) => handleInputChange('diabetes', e.target.value)}
                      className="input-field"
                    >
                      <option value="">Select</option>
                      <option value="1">Yes</option>
                      <option value="0">No</option>
                    </select>
                  </div>
                  
                  <div className="input-group">
                    <label className="input-label">Glucose (mg/dL)</label>
                    <input
                      type="number"
                      value={patientData.glucose}
                      onChange={(e) => handleInputChange('glucose', e.target.value)}
                      placeholder="e.g., 90"
                      className="input-field"
                      min="50"
                      max="500"
                    />
                  </div>
                  
                  <div className="input-group">
                    <label className="input-label">BP Medication</label>
                    <select
                      value={patientData.BPMeds}
                      onChange={(e) => handleInputChange('BPMeds', e.target.value)}
                      className="input-field"
                    >
                      <option value="">Select</option>
                      <option value="1">Yes</option>
                      <option value="0">No</option>
                    </select>
                  </div>
                </div>
                
                <div className="flex space-x-4 mt-6">
                  <button
                    onClick={predictHeartDisease}
                    disabled={loading}
                    className="btn-primary flex-1 flex items-center justify-center space-x-2"
                  >
                    {loading ? (
                      <>
                        <Loader className="h-4 w-4 animate-spin" />
                        <span>Analyzing...</span>
                      </>
                    ) : (
                      <>
                        <Heart className="h-4 w-4" />
                        <span>Predict Risk</span>
                      </>
                    )}
                  </button>
                  
                  <button
                    onClick={resetForm}
                    className="px-6 py-3 border border-slate-300 text-slate-700 rounded-lg hover:bg-slate-50 transition-colors"
                  >
                    Reset
                  </button>
                </div>
              </div>

              {/* Prediction Result */}
              <div className="form-section">
                <h3 className="subsection-title flex items-center space-x-2">
                  <Activity className="h-5 w-5 icon-medical" />
                  <span>Risk Assessment Result</span>
                </h3>
                
                {predictionResult ? (
                  <div className="space-y-4">
                    <div className={`prediction-result ${predictionResult.prediction === 1 ? 'prediction-high-risk' : 'prediction-low-risk'}`}>
                      <div className="flex items-center justify-center space-x-2 mb-2">
                        {predictionResult.prediction === 1 ? (
                          <AlertTriangle className="h-6 w-6" />
                        ) : (
                          <CheckCircle className="h-6 w-6" />
                        )}
                        <span>{predictionResult.interpretation}</span>
                      </div>
                      <div className="text-sm opacity-80">
                        Risk Probability: {predictionResult.risk_probability}%
                      </div>
                    </div>
                    
                    <div className="metric-card">
                      <div className="text-sm text-slate-600 mb-1">Model Used</div>
                      <div className="font-semibold text-slate-800">{predictionResult.model_used}</div>
                    </div>
                    
                    <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                      <p className="text-sm text-blue-800">
                        <strong>Note:</strong> This prediction is for informational purposes only. 
                        Please consult with a healthcare professional for proper medical advice and diagnosis.
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <Heart className="h-16 w-16 text-slate-300 mx-auto mb-4" />
                    <p className="text-slate-500">Enter patient information and click "Predict Risk" to see the assessment results.</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Feature Analysis Tab */}
        {activeTab === 'features' && (
          <div className="space-y-8">
            <div className="text-center">
              <h2 className="section-title">Feature Importance Analysis</h2>
              <p className="text-slate-600 max-w-2xl mx-auto">
                Discover which patient characteristics have the most impact on heart disease prediction using statistical analysis.
              </p>
            </div>

            <div className="flex justify-center">
              <button
                onClick={loadFeatureImportance}
                disabled={loading}
                className="btn-secondary flex items-center space-x-2"
              >
                {loading ? (
                  <>
                    <Loader className="h-4 w-4 animate-spin" />
                    <span>Analyzing Features...</span>
                  </>
                ) : (
                  <>
                    <Brain className="h-4 w-4" />
                    <span>Analyze Feature Importance</span>
                  </>
                )}
              </button>
            </div>

            {featureImportance.length > 0 && (
              <div className="grid lg:grid-cols-2 gap-8">
                <div className="chart-container">
                  <h3 className="subsection-title">Top 10 Most Important Features</h3>
                  <div className="chart-wrapper">
                    <Bar data={featureChart} options={{
                      ...chartOptions,
                      plugins: {
                        ...chartOptions.plugins,
                        title: {
                          ...chartOptions.plugins.title,
                          text: 'Feature Importance Scores'
                        }
                      }
                    }} />
                  </div>
                </div>
                
                <div className="feature-card">
                  <h3 className="subsection-title">Feature Rankings</h3>
                  <div className="space-y-3 max-h-96 overflow-y-auto">
                    {featureImportance.map((feature, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                        <div className="flex items-center space-x-3">
                          <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                            index < 3 ? 'bg-medical-100 text-medical-700' : 'bg-slate-200 text-slate-600'
                          }`}>
                            {index + 1}
                          </div>
                          <span className="font-medium text-slate-800">{feature.feature}</span>
                        </div>
                        <span className="text-sm text-slate-600 font-mono">
                          {feature.score.toFixed(2)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Model Comparison Tab */}
        {activeTab === 'models' && (
          <div className="space-y-8">
            <div className="text-center">
              <h2 className="section-title">Machine Learning Model Comparison</h2>
              <p className="text-slate-600 max-w-2xl mx-auto">
                Compare the performance of different AI algorithms to find the most accurate model for heart disease prediction.
              </p>
            </div>

            <div className="flex justify-center space-x-4">
              <button
                onClick={trainModels}
                disabled={loading}
                className="btn-primary flex items-center space-x-2"
              >
                {loading ? (
                  <>
                    <Loader className="h-4 w-4 animate-spin" />
                    <span>Training Models...</span>
                  </>
                ) : (
                  <>
                    <Brain className="h-4 w-4" />
                    <span>Train All Models</span>
                  </>
                )}
              </button>
              
              <button
                onClick={getModelComparison}
                disabled={loading}
                className="btn-secondary flex items-center space-x-2"
              >
                {loading ? (
                  <>
                    <Loader className="h-4 w-4 animate-spin" />
                    <span>Loading...</span>
                  </>
                ) : (
                  <>
                    <BarChart3 className="h-4 w-4" />
                    <span>View Comparison</span>
                  </>
                )}
              </button>
            </div>

            {bestModel && (
              <div className="text-center p-4 bg-health-50 border border-health-200 rounded-lg">
                <p className="text-health-800">
                  <strong>Best Performing Model:</strong> {bestModel}
                </p>
              </div>
            )}

            {modelResults.length > 0 && (
              <div className="grid lg:grid-cols-2 gap-8">
                <div className="chart-container">
                  <h3 className="subsection-title">Algorithm Performance Comparison</h3>
                  <div className="chart-wrapper">
                    <Bar data={modelComparisonChart} options={chartOptions} />
                  </div>
                </div>
                
                <div className="feature-card">
                  <h3 className="subsection-title">Detailed Model Metrics</h3>
                  <div className="space-y-4 max-h-96 overflow-y-auto">
                    {modelResults.map((result, index) => (
                      <div key={index} className={`p-4 rounded-lg border-2 ${
                        result.algorithm === bestModel 
                          ? 'bg-health-50 border-health-200' 
                          : 'bg-white border-slate-200'
                      }`}>
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="font-semibold text-slate-800">{result.algorithm}</h4>
                          {result.algorithm === bestModel && (
                            <span className="px-2 py-1 bg-health-100 text-health-700 text-xs rounded-full font-medium">
                              Best
                            </span>
                          )}
                        </div>
                        
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div>
                            <span className="text-slate-600">Accuracy:</span>
                            <span className="font-medium ml-1">{result.accuracy}%</span>
                          </div>
                          <div>
                            <span className="text-slate-600">F1-Score:</span>
                            <span className="font-medium ml-1">{result.f1_score}%</span>
                          </div>
                          <div>
                            <span className="text-slate-600">Precision:</span>
                            <span className="font-medium ml-1">{result.precision}%</span>
                          </div>
                          <div>
                            <span className="text-slate-600">Recall:</span>
                            <span className="font-medium ml-1">{result.recall}%</span>
                          </div>
                        </div>
                        
                        {result.roc_auc > 0 && (
                          <div className="mt-2 text-sm">
                            <span className="text-slate-600">ROC-AUC:</span>
                            <span className="font-medium ml-1">{result.roc_auc}</span>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;