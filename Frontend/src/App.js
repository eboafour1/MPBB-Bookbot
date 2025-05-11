import React, { useState, useMemo } from 'react';
import {
  ThemeProvider,
  CssBaseline,
  Container,
  Box,
  Button,
  Grid
} from '@mui/material';
import axios from 'axios';
import getTheme from './theme';
import NavBar from './components/NavBar';
import Header from './components/Header';
import FileUpload from './components/FileUpload';
import ModelSelector from './components/ModelSelector';
import SummaryLengthSelector from './components/SummaryLengthSelector';
import TextInput from './components/TextInput';
import SummaryDisplay from './components/SummaryDisplay';
import Notification from './components/Notification';
import Footer from './components/Footer';

function App() {
  const [mode, setMode] = useState('light');
  const theme = useMemo(() => getTheme(mode), [mode]);
  const toggleMode = () => setMode(prev => (prev === 'light' ? 'dark' : 'light'));

  const [text, setText] = useState('');
  const [model, setModel] = useState('pegasus');
  const [length, setLength] = useState('medium');
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);
  const [notif, setNotif] = useState({ open: false, message: '', severity: 'info' });

  const handleSummarize = async () => {
    if (!text) {
      setNotif({ open: true, message: 'Please provide text or upload a file.', severity: 'warning' });
      return;
    }
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/api/summarize', {
        text,
        model_choice: model,
        summary_length: length
      });
      setSummary(response.data.summary);
      setNotif({ open: true, message: 'Summary generated successfully!', severity: 'success' });
    } catch (err) {
      console.error(err);
      setNotif({ open: true, message: 'Failed to generate summary.', severity: 'error' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <NavBar mode={mode} toggleMode={toggleMode} />
      <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
        <Header />
        <Box mt={4}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FileUpload onFileRead={setText} />
            </Grid>
            <Grid item xs={12} md={6}>
              <ModelSelector model={model} setModel={setModel} />
              <SummaryLengthSelector length={length} setLength={setLength} />
            </Grid>
            <Grid item xs={12}>
              <TextInput text={text} setText={setText} />
            </Grid>
            <Grid item xs={12}>
              <Button
                variant="contained"
                color="primary"
                onClick={handleSummarize}
                disabled={loading}
                fullWidth
              >
                {loading ? 'Summarizing...' : 'Summarize'}
              </Button>
            </Grid>
            <Grid item xs={12}>
              <SummaryDisplay summary={summary} />
            </Grid>
          </Grid>
        </Box>
      </Container>
      <Footer />
      <Notification
        open={notif.open}
        message={notif.message}
        severity={notif.severity}
        onClose={() => setNotif({ ...notif, open: false })}
      />
    </ThemeProvider>
  );
}

export default App;