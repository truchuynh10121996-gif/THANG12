/**
 * ModelTraining Page - Training và đánh giá models
 * ==================================================
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Typography,
  Card,
  CardContent,
  Button,
  Alert,
  CircularProgress,
  Chip,
  LinearProgress,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Check as CheckIcon,
  Close as CloseIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { getModelStatus, trainLayer1, trainLayer2, trainAll, getMetrics } from '../services/api';

function ModelTraining() {
  const [modelStatus, setModelStatus] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);
  const [trainingModel, setTrainingModel] = useState(null);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [statusRes, metricsRes] = await Promise.all([
        getModelStatus(),
        getMetrics(),
      ]);

      setModelStatus(statusRes.status);
      setMetrics(metricsRes.metrics);
      setError(null);
    } catch (err) {
      setError('Không thể kết nối đến ML Service');
      // Dữ liệu mẫu
      setModelStatus({
        layer1: { fitted: true, models: ['isolation_forest', 'lightgbm'] },
        layer2: {
          fitted: true,
          models: { autoencoder: true, lstm: true, gnn: false },
        },
        weights: { layer1: 0.4, layer2: 0.6 },
      });
      setMetrics({
        isolation_forest: { precision: 0.75, recall: 0.80, f1_score: 0.77, roc_auc: 0.88 },
        lightgbm: { precision: 0.85, recall: 0.78, f1_score: 0.81, roc_auc: 0.92 },
        ensemble: { precision: 0.83, recall: 0.80, f1_score: 0.81, roc_auc: 0.93 },
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleTrain = async (layer) => {
    try {
      setTraining(true);
      setTrainingModel(layer);
      setSuccess(null);
      setError(null);

      let result;
      if (layer === 'layer1') {
        result = await trainLayer1();
      } else if (layer === 'layer2') {
        result = await trainLayer2();
      } else {
        result = await trainAll();
      }

      setSuccess(`Training ${layer} hoàn tất!`);
      await fetchData(); // Refresh data
    } catch (err) {
      setError(`Training ${layer} thất bại: ${err.message}`);
    } finally {
      setTraining(false);
      setTrainingModel(null);
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h5" sx={{ fontWeight: 600 }}>
          Training Models
        </Typography>
        <Button startIcon={<RefreshIcon />} onClick={fetchData}>
          Refresh
        </Button>
      </Box>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      {success && <Alert severity="success" sx={{ mb: 2 }}>{success}</Alert>}

      <Grid container spacing={3}>
        {/* Layer 1 */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>
                Layer 1: Global Fraud Detection
              </Typography>

              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  Models:
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                  <Chip
                    icon={modelStatus?.layer1?.fitted ? <CheckIcon /> : <CloseIcon />}
                    label="Isolation Forest"
                    color={modelStatus?.layer1?.fitted ? 'success' : 'default'}
                    size="small"
                  />
                  <Chip
                    icon={modelStatus?.layer1?.fitted ? <CheckIcon /> : <CloseIcon />}
                    label="LightGBM"
                    color={modelStatus?.layer1?.fitted ? 'success' : 'default'}
                    size="small"
                  />
                </Box>
              </Box>

              <Button
                variant="contained"
                startIcon={
                  training && trainingModel === 'layer1' ? (
                    <CircularProgress size={20} color="inherit" />
                  ) : (
                    <PlayIcon />
                  )
                }
                onClick={() => handleTrain('layer1')}
                disabled={training}
                fullWidth
              >
                {training && trainingModel === 'layer1' ? 'Đang training...' : 'Train Layer 1'}
              </Button>

              {training && trainingModel === 'layer1' && (
                <LinearProgress sx={{ mt: 2 }} />
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Layer 2 */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>
                Layer 2: User Profile (Advanced)
              </Typography>

              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  Models:
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                  <Chip
                    icon={modelStatus?.layer2?.models?.autoencoder ? <CheckIcon /> : <CloseIcon />}
                    label="Autoencoder"
                    color={modelStatus?.layer2?.models?.autoencoder ? 'success' : 'default'}
                    size="small"
                  />
                  <Chip
                    icon={modelStatus?.layer2?.models?.lstm ? <CheckIcon /> : <CloseIcon />}
                    label="LSTM"
                    color={modelStatus?.layer2?.models?.lstm ? 'success' : 'default'}
                    size="small"
                  />
                  <Chip
                    icon={modelStatus?.layer2?.models?.gnn ? <CheckIcon /> : <CloseIcon />}
                    label="GNN"
                    color={modelStatus?.layer2?.models?.gnn ? 'success' : 'default'}
                    size="small"
                  />
                </Box>
              </Box>

              <Button
                variant="contained"
                color="secondary"
                startIcon={
                  training && trainingModel === 'layer2' ? (
                    <CircularProgress size={20} color="inherit" />
                  ) : (
                    <PlayIcon />
                  )
                }
                onClick={() => handleTrain('layer2')}
                disabled={training}
                fullWidth
              >
                {training && trainingModel === 'layer2' ? 'Đang training...' : 'Train Layer 2'}
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {/* Train All */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Button
                variant="outlined"
                size="large"
                startIcon={
                  training && trainingModel === 'all' ? (
                    <CircularProgress size={20} />
                  ) : (
                    <PlayIcon />
                  )
                }
                onClick={() => handleTrain('all')}
                disabled={training}
                fullWidth
              >
                {training && trainingModel === 'all' ? 'Đang training tất cả...' : 'Train Tất cả Models'}
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {/* Metrics */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>
                Metrics đánh giá
              </Typography>

              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Model</TableCell>
                      <TableCell align="right">Precision</TableCell>
                      <TableCell align="right">Recall</TableCell>
                      <TableCell align="right">F1-Score</TableCell>
                      <TableCell align="right">ROC-AUC</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {metrics && Object.entries(metrics).map(([name, m]) => (
                      <TableRow key={name}>
                        <TableCell component="th" scope="row">
                          {name.replace('_', ' ').toUpperCase()}
                        </TableCell>
                        <TableCell align="right">{(m.precision * 100).toFixed(1)}%</TableCell>
                        <TableCell align="right">{(m.recall * 100).toFixed(1)}%</TableCell>
                        <TableCell align="right">{(m.f1_score * 100).toFixed(1)}%</TableCell>
                        <TableCell align="right">{(m.roc_auc * 100).toFixed(1)}%</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default ModelTraining;
