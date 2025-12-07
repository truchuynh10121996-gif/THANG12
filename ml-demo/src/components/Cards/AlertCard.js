/**
 * AlertCard Component - Card hiển thị cảnh báo
 * =============================================
 */

import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  IconButton,
  Collapse,
} from '@mui/material';
import {
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  ExpandMore as ExpandIcon,
  CheckCircle as CheckIcon,
} from '@mui/icons-material';
import { riskColors } from '../../styles/theme';

function AlertCard({
  title,
  description,
  riskLevel,
  timestamp,
  transactionId,
  amount,
  userId,
  expanded = false,
  onToggle,
}) {
  // Icon theo risk level
  const getIcon = () => {
    switch (riskLevel) {
      case 'critical':
        return <ErrorIcon sx={{ color: riskColors.critical }} />;
      case 'high':
        return <WarningIcon sx={{ color: riskColors.high }} />;
      case 'medium':
        return <InfoIcon sx={{ color: riskColors.medium }} />;
      default:
        return <CheckIcon sx={{ color: riskColors.low }} />;
    }
  };

  // Label theo risk level
  const getRiskLabel = () => {
    switch (riskLevel) {
      case 'critical':
        return 'Nghiêm trọng';
      case 'high':
        return 'Cao';
      case 'medium':
        return 'Trung bình';
      default:
        return 'Thấp';
    }
  };

  return (
    <Card
      sx={{
        mb: 1,
        borderLeft: `4px solid ${riskColors[riskLevel] || riskColors.low}`,
      }}
    >
      <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {getIcon()}

          <Box sx={{ flexGrow: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
              <Typography variant="body1" sx={{ fontWeight: 500 }}>
                {title}
              </Typography>
              <Chip
                label={getRiskLabel()}
                size="small"
                sx={{
                  backgroundColor: riskColors[riskLevel] || riskColors.low,
                  color: '#fff',
                  fontWeight: 500,
                  fontSize: '0.7rem',
                  height: 20,
                }}
              />
            </Box>

            <Typography variant="body2" color="text.secondary">
              {description}
            </Typography>
          </Box>

          <Box sx={{ textAlign: 'right' }}>
            <Typography variant="caption" color="text.secondary">
              {timestamp}
            </Typography>
            {onToggle && (
              <IconButton size="small" onClick={onToggle}>
                <ExpandIcon
                  sx={{
                    transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
                    transition: 'transform 0.2s',
                  }}
                />
              </IconButton>
            )}
          </Box>
        </Box>

        {/* Chi tiết mở rộng */}
        <Collapse in={expanded}>
          <Box
            sx={{
              mt: 2,
              pt: 2,
              borderTop: '1px solid',
              borderColor: 'divider',
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              gap: 2,
            }}
          >
            <Box>
              <Typography variant="caption" color="text.secondary">
                Transaction ID
              </Typography>
              <Typography variant="body2">{transactionId}</Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Số tiền
              </Typography>
              <Typography variant="body2">
                {amount?.toLocaleString()} VND
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">
                User ID
              </Typography>
              <Typography variant="body2">{userId}</Typography>
            </Box>
          </Box>
        </Collapse>
      </CardContent>
    </Card>
  );
}

export default AlertCard;
