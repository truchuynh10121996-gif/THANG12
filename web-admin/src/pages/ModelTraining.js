import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Button,
  Card,
  CardContent,
  CardActions,
  Divider,
  Alert,
  AlertTitle,
  CircularProgress,
  Chip,
  Stack,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Slider
} from '@mui/material';
import {
  PlayArrow,
  CheckCircle,
  Error,
  Warning,
  ExpandMore,
  Hub,
  Psychology,
  Timeline,
  Tune,
  Refresh,
  Speed,
  AccountTree,
  Security,
  Analytics
} from '@mui/icons-material';
import toast from 'react-hot-toast';
import api from '../services/api';

// ==================== COMPONENT: GNN TRAINING SECTION ====================
const GNNTrainingSection = () => {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [buildLoading, setBuildLoading] = useState(false);
  const [trainLoading, setTrainLoading] = useState(false);
  const [buildResult, setBuildResult] = useState(null);
  const [trainResult, setTrainResult] = useState(null);
  const [configOpen, setConfigOpen] = useState(false);
  const [trainConfig, setTrainConfig] = useState({
    epochs: 100,
    hidden_channels: 64,
    num_layers: 2,
    learning_rate: 0.001,
    dropout: 0.3
  });

  // Load GNN status
  const loadStatus = useCallback(async () => {
    setLoading(true);
    try {
      const response = await api.get('/train/gnn/status');
      if (response.data.success) {
        setStatus(response.data.status);
      }
    } catch (error) {
      console.error('Failed to load GNN status:', error);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    loadStatus();
  }, [loadStatus]);

  // STEP 1: Build GNN Graph
  const handleBuildGraph = async () => {
    setBuildLoading(true);
    setBuildResult(null);

    try {
      toast.loading('Dang tao mang luoi GNN...', { id: 'gnn-build' });

      const response = await api.post('/train/gnn/build');

      if (response.data.success) {
        toast.success('Tao mang luoi GNN thanh cong!', { id: 'gnn-build' });
        setBuildResult(response.data);
        loadStatus(); // Refresh status
      } else {
        toast.error(response.data.error || 'Tao mang luoi that bai', { id: 'gnn-build' });
        setBuildResult({ success: false, error: response.data.error });
      }
    } catch (error) {
      const errorMsg = error.response?.data?.error || error.message;
      toast.error(`Loi: ${errorMsg}`, { id: 'gnn-build' });
      setBuildResult({ success: false, error: errorMsg });
    }

    setBuildLoading(false);
  };

  // STEP 2: Train GNN Model
  const handleTrainModel = async () => {
    setTrainLoading(true);
    setTrainResult(null);

    try {
      toast.loading('Dang huan luyen GNN...', { id: 'gnn-train' });

      const response = await api.post('/train/gnn/train', trainConfig);

      if (response.data.success) {
        toast.success('Huan luyen GNN thanh cong!', { id: 'gnn-train' });
        setTrainResult(response.data);
        loadStatus(); // Refresh status
      } else {
        toast.error(response.data.error || 'Huan luyen that bai', { id: 'gnn-train' });
        setTrainResult({ success: false, error: response.data.error });
      }
    } catch (error) {
      const errorMsg = error.response?.data?.error || error.message;
      toast.error(`Loi: ${errorMsg}`, { id: 'gnn-train' });
      setTrainResult({ success: false, error: errorMsg });
    }

    setTrainLoading(false);
  };

  return (
    <Paper sx={{ p: 3, mb: 3, borderRadius: 3, boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <Hub sx={{ fontSize: 40, color: '#9C27B0', mr: 2 }} />
        <Box>
          <Typography variant="h5" sx={{ fontWeight: 700, color: '#9C27B0' }}>
            GNN - Graph Neural Network
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Heterogeneous Graph cho Edge-Level Fraud Detection
          </Typography>
        </Box>
        <Box sx={{ flexGrow: 1 }} />
        <Button
          startIcon={<Refresh />}
          onClick={loadStatus}
          disabled={loading}
          sx={{ mr: 1 }}
        >
          Refresh
        </Button>
      </Box>

      {/* Status Cards */}
      {status && (
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={4}>
            <Card sx={{
              bgcolor: status.data_exists ? '#E8F5E9' : '#FFEBEE',
              height: '100%'
            }}>
              <CardContent>
                <Stack direction="row" alignItems="center" spacing={1}>
                  {status.data_exists ?
                    <CheckCircle sx={{ color: '#4CAF50' }} /> :
                    <Error sx={{ color: '#F44336' }} />
                  }
                  <Typography variant="subtitle2">Du lieu GNN</Typography>
                </Stack>
                <Typography variant="body2" sx={{ mt: 1 }}>
                  {status.data_exists ? 'Du lieu da san sang' : 'Chua co du lieu'}
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} sm={4}>
            <Card sx={{
              bgcolor: status.graph_ready ? '#E8F5E9' : '#FFF3E0',
              height: '100%'
            }}>
              <CardContent>
                <Stack direction="row" alignItems="center" spacing={1}>
                  {status.graph_ready ?
                    <CheckCircle sx={{ color: '#4CAF50' }} /> :
                    <Warning sx={{ color: '#FF9800' }} />
                  }
                  <Typography variant="subtitle2">Mang luoi Graph</Typography>
                </Stack>
                <Typography variant="body2" sx={{ mt: 1 }}>
                  {status.graph_ready ? 'Graph da duoc tao' : 'Chua tao graph'}
                </Typography>
                {status.graph_stats && (
                  <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
                    {status.graph_stats.num_users} users, {status.graph_stats.num_transfer_edges} edges
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} sm={4}>
            <Card sx={{
              bgcolor: status.model_ready ? '#E8F5E9' : '#FFF3E0',
              height: '100%'
            }}>
              <CardContent>
                <Stack direction="row" alignItems="center" spacing={1}>
                  {status.model_ready ?
                    <CheckCircle sx={{ color: '#4CAF50' }} /> :
                    <Warning sx={{ color: '#FF9800' }} />
                  }
                  <Typography variant="subtitle2">Model GNN</Typography>
                </Stack>
                <Typography variant="body2" sx={{ mt: 1 }}>
                  {status.model_ready ? 'Model da huan luyen' : 'Chua huan luyen'}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      <Divider sx={{ my: 3 }} />

      {/* TWO BUTTONS SECTION */}
      <Grid container spacing={3}>
        {/* STEP 1: Build Graph */}
        <Grid item xs={12} md={6}>
          <Card
            sx={{
              height: '100%',
              border: '2px solid #9C27B0',
              borderRadius: 3,
              transition: 'transform 0.2s',
              '&:hover': { transform: 'translateY(-4px)' }
            }}
          >
            <CardContent>
              <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 2 }}>
                <AccountTree sx={{ fontSize: 36, color: '#9C27B0' }} />
                <Box>
                  <Typography variant="h6" sx={{ fontWeight: 700 }}>
                    Buoc 1: Tao Mang Luoi GNN
                  </Typography>
                  <Chip label="Build Graph" size="small" color="secondary" />
                </Box>
              </Stack>

              <Typography variant="body2" sx={{ mb: 2, color: 'text.secondary' }}>
                Load du lieu tu gnn_data/, kiem tra tinh toan ven (Sanity Check),
                xay dung Heterogeneous Graph va luu ra file.
              </Typography>

              <Box sx={{ p: 2, bgcolor: '#F3E5F5', borderRadius: 2, mb: 2 }}>
                <Typography variant="caption" sx={{ fontWeight: 600 }}>
                  Thuc hien:
                </Typography>
                <ul style={{ margin: '8px 0', paddingLeft: 20 }}>
                  <li><Typography variant="caption">Load nodes.csv, edges_*.csv, labels, splits</Typography></li>
                  <li><Typography variant="caption">Sanity Check - Kiem tra du lieu</Typography></li>
                  <li><Typography variant="caption">Build PyTorch Geometric HeteroData</Typography></li>
                  <li><Typography variant="caption">Luu graph va tao flag file</Typography></li>
                </ul>
              </Box>

              {/* Build Result */}
              {buildResult && (
                <Alert
                  severity={buildResult.success ? 'success' : 'error'}
                  sx={{ mb: 2 }}
                >
                  {buildResult.success ? (
                    <>
                      <AlertTitle>Thanh cong!</AlertTitle>
                      <Typography variant="body2">
                        Graph: {buildResult.stats?.num_users} users, {buildResult.stats?.num_transfer_edges} edges
                      </Typography>
                      <Typography variant="body2">
                        Fraud: {buildResult.stats?.num_fraud} ({(buildResult.stats?.fraud_ratio * 100)?.toFixed(1)}%)
                      </Typography>
                    </>
                  ) : (
                    <>
                      <AlertTitle>Loi</AlertTitle>
                      {buildResult.error}
                    </>
                  )}
                </Alert>
              )}
            </CardContent>

            <CardActions sx={{ p: 2, pt: 0 }}>
              <Button
                fullWidth
                variant="contained"
                size="large"
                startIcon={buildLoading ? <CircularProgress size={20} color="inherit" /> : <AccountTree />}
                onClick={handleBuildGraph}
                disabled={buildLoading || !status?.data_exists}
                sx={{
                  background: 'linear-gradient(135deg, #9C27B0 0%, #7B1FA2 100%)',
                  py: 1.5,
                  fontSize: '1rem',
                  fontWeight: 700
                }}
              >
                {buildLoading ? 'Dang tao...' : 'Tao Mang Luoi GNN'}
              </Button>
            </CardActions>
          </Card>
        </Grid>

        {/* STEP 2: Train Model */}
        <Grid item xs={12} md={6}>
          <Card
            sx={{
              height: '100%',
              border: '2px solid #4CAF50',
              borderRadius: 3,
              transition: 'transform 0.2s',
              '&:hover': { transform: 'translateY(-4px)' },
              opacity: status?.graph_ready ? 1 : 0.6
            }}
          >
            <CardContent>
              <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 2 }}>
                <Psychology sx={{ fontSize: 36, color: '#4CAF50' }} />
                <Box>
                  <Typography variant="h6" sx={{ fontWeight: 700 }}>
                    Buoc 2: Huan Luyen GNN
                  </Typography>
                  <Chip label="Train Model" size="small" color="success" />
                </Box>
              </Stack>

              <Typography variant="body2" sx={{ mb: 2, color: 'text.secondary' }}>
                Load graph da build, train HeteroGNN Model,
                danh gia tren test set va luu model.
              </Typography>

              {!status?.graph_ready && (
                <Alert severity="warning" sx={{ mb: 2 }}>
                  Vui long chay Buoc 1 truoc!
                </Alert>
              )}

              <Box sx={{ p: 2, bgcolor: '#E8F5E9', borderRadius: 2, mb: 2 }}>
                <Stack direction="row" justifyContent="space-between" alignItems="center">
                  <Typography variant="caption" sx={{ fontWeight: 600 }}>
                    Cau hinh Training:
                  </Typography>
                  <Button
                    size="small"
                    startIcon={<Tune />}
                    onClick={() => setConfigOpen(true)}
                  >
                    Chinh sua
                  </Button>
                </Stack>
                <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                  Epochs: {trainConfig.epochs}, Hidden: {trainConfig.hidden_channels},
                  Layers: {trainConfig.num_layers}, LR: {trainConfig.learning_rate}
                </Typography>
              </Box>

              {/* Train Result */}
              {trainResult && (
                <Alert
                  severity={trainResult.success ? 'success' : 'error'}
                  sx={{ mb: 2 }}
                >
                  {trainResult.success ? (
                    <>
                      <AlertTitle>Huan luyen thanh cong!</AlertTitle>
                      <Typography variant="body2">
                        Accuracy: {(trainResult.metrics?.accuracy * 100)?.toFixed(2)}%
                      </Typography>
                      <Typography variant="body2">
                        F1-Score: {(trainResult.metrics?.f1_score * 100)?.toFixed(2)}%
                      </Typography>
                      <Typography variant="body2">
                        ROC-AUC: {(trainResult.metrics?.roc_auc * 100)?.toFixed(2)}%
                      </Typography>
                    </>
                  ) : (
                    <>
                      <AlertTitle>Loi</AlertTitle>
                      {trainResult.error}
                    </>
                  )}
                </Alert>
              )}
            </CardContent>

            <CardActions sx={{ p: 2, pt: 0 }}>
              <Button
                fullWidth
                variant="contained"
                size="large"
                startIcon={trainLoading ? <CircularProgress size={20} color="inherit" /> : <PlayArrow />}
                onClick={handleTrainModel}
                disabled={trainLoading || !status?.graph_ready}
                sx={{
                  background: 'linear-gradient(135deg, #4CAF50 0%, #388E3C 100%)',
                  py: 1.5,
                  fontSize: '1rem',
                  fontWeight: 700
                }}
              >
                {trainLoading ? 'Dang huan luyen...' : 'Huan Luyen GNN'}
              </Button>
            </CardActions>
          </Card>
        </Grid>
      </Grid>

      {/* Training Config Dialog */}
      <Dialog open={configOpen} onClose={() => setConfigOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle sx={{ bgcolor: '#4CAF50', color: '#fff' }}>
          Cau hinh Training GNN
        </DialogTitle>
        <DialogContent sx={{ pt: 3 }}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography gutterBottom>Epochs: {trainConfig.epochs}</Typography>
              <Slider
                value={trainConfig.epochs}
                onChange={(e, v) => setTrainConfig({...trainConfig, epochs: v})}
                min={10}
                max={500}
                step={10}
                marks={[
                  { value: 50, label: '50' },
                  { value: 100, label: '100' },
                  { value: 200, label: '200' },
                  { value: 500, label: '500' }
                ]}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Hidden Channels"
                type="number"
                value={trainConfig.hidden_channels}
                onChange={(e) => setTrainConfig({...trainConfig, hidden_channels: parseInt(e.target.value)})}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Num Layers"
                type="number"
                value={trainConfig.num_layers}
                onChange={(e) => setTrainConfig({...trainConfig, num_layers: parseInt(e.target.value)})}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Learning Rate"
                type="number"
                inputProps={{ step: 0.0001 }}
                value={trainConfig.learning_rate}
                onChange={(e) => setTrainConfig({...trainConfig, learning_rate: parseFloat(e.target.value)})}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Dropout"
                type="number"
                inputProps={{ step: 0.1, min: 0, max: 0.9 }}
                value={trainConfig.dropout}
                onChange={(e) => setTrainConfig({...trainConfig, dropout: parseFloat(e.target.value)})}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfigOpen(false)}>Dong</Button>
          <Button
            variant="contained"
            onClick={() => setConfigOpen(false)}
            sx={{ bgcolor: '#4CAF50' }}
          >
            Luu
          </Button>
        </DialogActions>
      </Dialog>
    </Paper>
  );
};

// ==================== COMPONENT: OTHER MODELS SECTION ====================
const OtherModelsSection = () => {
  const [loading, setLoading] = useState({});
  const [results, setResults] = useState({});

  const models = [
    {
      id: 'isolation_forest',
      name: 'Isolation Forest',
      description: 'Phat hien anomaly khong can label (Unsupervised)',
      icon: <Security sx={{ fontSize: 32, color: '#2196F3' }} />,
      color: '#2196F3',
      endpoint: '/train/isolation_forest'
    },
    {
      id: 'lightgbm',
      name: 'LightGBM',
      description: 'Gradient Boosting cho classification (Supervised)',
      icon: <Speed sx={{ fontSize: 32, color: '#FF9800' }} />,
      color: '#FF9800',
      endpoint: '/train/lightgbm'
    },
    {
      id: 'autoencoder',
      name: 'Autoencoder',
      description: 'Neural network cho anomaly detection',
      icon: <Analytics sx={{ fontSize: 32, color: '#E91E63' }} />,
      color: '#E91E63',
      endpoint: '/train/autoencoder'
    },
    {
      id: 'lstm',
      name: 'LSTM',
      description: 'Sequence model cho temporal patterns',
      icon: <Timeline sx={{ fontSize: 32, color: '#00BCD4' }} />,
      color: '#00BCD4',
      endpoint: '/train/lstm'
    }
  ];

  const handleFileUpload = async (modelId, endpoint, file) => {
    if (!file) {
      toast.error('Vui long chon file CSV');
      return;
    }

    setLoading(prev => ({ ...prev, [modelId]: true }));

    try {
      const formData = new FormData();
      formData.append('file', file);

      toast.loading(`Dang train ${modelId}...`, { id: `train-${modelId}` });

      const response = await api.post(endpoint, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      if (response.data.success) {
        toast.success(`Train ${modelId} thanh cong!`, { id: `train-${modelId}` });
        setResults(prev => ({ ...prev, [modelId]: response.data }));
      } else {
        toast.error(response.data.error, { id: `train-${modelId}` });
        setResults(prev => ({ ...prev, [modelId]: { error: response.data.error } }));
      }
    } catch (error) {
      toast.error(`Loi: ${error.message}`, { id: `train-${modelId}` });
      setResults(prev => ({ ...prev, [modelId]: { error: error.message } }));
    }

    setLoading(prev => ({ ...prev, [modelId]: false }));
  };

  return (
    <Paper sx={{ p: 3, borderRadius: 3, boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
      <Typography variant="h5" sx={{ fontWeight: 700, color: '#FF8DAD', mb: 3 }}>
        Cac Model Khac
      </Typography>

      <Grid container spacing={3}>
        {models.map((model) => (
          <Grid item xs={12} sm={6} md={3} key={model.id}>
            <Card sx={{
              height: '100%',
              borderTop: `4px solid ${model.color}`,
              transition: 'transform 0.2s',
              '&:hover': { transform: 'translateY(-4px)' }
            }}>
              <CardContent>
                <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 2 }}>
                  {model.icon}
                  <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                    {model.name}
                  </Typography>
                </Stack>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2, minHeight: 40 }}>
                  {model.description}
                </Typography>

                {results[model.id]?.metrics && (
                  <Box sx={{ mt: 1 }}>
                    <Chip
                      label={`AUC: ${(results[model.id].metrics.roc_auc * 100)?.toFixed(1)}%`}
                      size="small"
                      color="success"
                    />
                  </Box>
                )}
              </CardContent>
              <CardActions sx={{ p: 2, pt: 0 }}>
                <Button
                  component="label"
                  fullWidth
                  variant="outlined"
                  disabled={loading[model.id]}
                  startIcon={loading[model.id] ? <CircularProgress size={16} /> : <PlayArrow />}
                  sx={{ borderColor: model.color, color: model.color }}
                >
                  {loading[model.id] ? 'Training...' : 'Upload & Train'}
                  <input
                    type="file"
                    hidden
                    accept=".csv"
                    onChange={(e) => handleFileUpload(model.id, model.endpoint, e.target.files[0])}
                  />
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Paper>
  );
};

// ==================== MAIN COMPONENT ====================
export default function ModelTraining() {
  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 3, fontWeight: 700, color: '#FF8DAD' }}>
        Huan Luyen Mo Hinh
      </Typography>

      <Alert severity="info" sx={{ mb: 3 }}>
        <AlertTitle>Huong dan</AlertTitle>
        <Typography variant="body2">
          <strong>GNN (Graph Neural Network):</strong> Su dung 2 buoc rieng biet -
          Buoc 1 tao mang luoi graph, Buoc 2 huan luyen model. Du lieu GNN nam trong thu muc gnn_data/.
        </Typography>
        <Typography variant="body2" sx={{ mt: 1 }}>
          <strong>Cac model khac:</strong> Upload file CSV de train tung model rieng le.
        </Typography>
      </Alert>

      {/* GNN Training Section - 2 BUTTONS */}
      <GNNTrainingSection />

      {/* Other Models Section */}
      <OtherModelsSection />
    </Box>
  );
}
