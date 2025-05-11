// File: src/App.js
import React, { useState, useMemo, useRef } from 'react';
import { CssBaseline, Container, Box, Button, Grid, LinearProgress, Typography } from '@mui/material';
import { ThemeProvider } from '@mui/material/styles';
import axios from 'axios';
import getTheme from './theme';
import NavBar from './components/NavBar';
import Header from './components/Header';
import FileUpload from './components/FileUpload';
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
  const [length, setLength] = useState('medium');
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState('');
  const [notif, setNotif] = useState({ open: false, message: '', severity: 'info' });
  const timerRef = useRef(null);

  const handleSummarize = async (inputText) => {
    setLoading(true);
    setProgress(0);
    setStatusMsg('Initializing summary…');
    timerRef.current = setInterval(() => {
      setProgress(prev => {
        const next = Math.min(prev + Math.random() * 10, 80);
        if (next > 50) setStatusMsg('Generating summary…');
        return next;
      });
    }, 500);

    try {
      const { data } = await axios.post(
        'http://localhost:8000/api/summarize/',
        { text: inputText, summary_length: length }
      );
      clearInterval(timerRef.current);
      setProgress(100);
      setStatusMsg('Finalizing…');
      await new Promise(res => setTimeout(res, 300));
      setSummary(data.summary);
      setNotif({ open: true, message: 'Summary generated successfully!', severity: 'success' });
    } catch (err) {
      clearInterval(timerRef.current);
      console.error(err);
      setNotif({ open: true, message: 'Failed to generate summary.', severity: 'error' });
    } finally {
      setLoading(false);
      setTimeout(() => setProgress(0), 300);
    }
  };

  const handleFileSummarize = async (file) => {
    setLoading(true);
    setProgress(0);
    setStatusMsg('Uploading file…');
    const formData = new FormData();
    formData.append('file', file);
    formData.append('summary_length', length);

    try {
      const { data } = await axios.post(
        'http://localhost:8000/api/summarize/file',
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: evt => {
            const pct = Math.round((evt.loaded * 100) / evt.total);
            setProgress(pct * 0.5);
            setStatusMsg(`Uploading… ${pct}%`);
          }
        }
      );
      setProgress(80);
      setStatusMsg('Summarizing file…');
      await new Promise(res => setTimeout(res, 500));
      setSummary(data.summary);
      setNotif({ open: true, message: 'Summary generated!', severity: 'success' });
    } catch (err) {
      console.error(err);
      setNotif({ open: true, message: 'File summarization failed.', severity: 'error' });
    } finally {
      setLoading(false);
      setTimeout(() => setProgress(0), 300);
    }
  };

  const handleFrontText = (parsedText, file) => {
    if (parsedText) setText(parsedText);
    else if (file) handleFileSummarize(file);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <NavBar mode={mode} toggleMode={toggleMode} />
      <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
        <Header />

        {loading && (
          <Box sx={{ width: '100%', mb: 2 }}>
            <Typography variant="body2" align="center">{statusMsg}</Typography>
            <LinearProgress variant="determinate" value={progress} />
          </Box>
        )}

        <Box mt={4}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FileUpload onText={handleFrontText} />
            </Grid>
            <Grid item xs={12} md={6}>
              <SummaryLengthSelector
                length={length}
                setLength={setLength}
                options={[
                  { value: 'short', label: 'Short' },
                  { value: 'medium', label: 'Moderate' },
                  { value: 'detailed', label: 'Detailed' }
                ]}
              />
            </Grid>
            <Grid item xs={12}>
              <TextInput text={text} setText={setText} rows={12} />
            </Grid>
            <Grid item xs={12}>
              <Button
                variant="contained"
                color="primary"
                onClick={() => handleSummarize(text)}
                disabled={loading}
                fullWidth
              >
                {loading ? 'Summarizing…' : 'Summarize'}
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
