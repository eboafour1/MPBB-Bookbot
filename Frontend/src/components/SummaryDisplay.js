import React from 'react';
import { Paper, Box, Typography, IconButton, Tooltip } from '@mui/material';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';

const SummaryDisplay = ({ summary }) => {
  const handleDownload = () => {
    const blob = new Blob([summary], { type: 'text/plain' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'summary.txt';
    link.click();
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(summary);
  };

  if (!summary) return null;

  return (
    <Paper elevation={3} sx={{ p: 2, position: 'relative' }}>
      <Typography variant="h5" gutterBottom>Summary</Typography>
      <Typography variant="body1" sx={{ whiteSpace: 'pre-line' }}>{summary}</Typography>
      <Box sx={{ position: 'absolute', top: 8, right: 8 }}>
        <Tooltip title="Download"> <IconButton onClick={handleDownload}><FileDownloadIcon /></IconButton> </Tooltip>
        <Tooltip title="Copy to clipboard"> <IconButton onClick={handleCopy}><ContentCopyIcon /></IconButton> </Tooltip>
      </Box>
    </Paper>
  );
};

export default SummaryDisplay;