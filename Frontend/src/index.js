import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import getTheme from './theme';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { GlobalWorkerOptions } from 'pdfjs-dist';
GlobalWorkerOptions.workerSrc = `${process.env.PUBLIC_URL}/pdf.worker.min.js`;


// 2) Create React 18 root and mount your app
const container = document.getElementById('root');
const root = ReactDOM.createRoot(container);

root.render(
  <React.StrictMode>
    <ThemeProvider theme={getTheme('light')}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  </React.StrictMode>
);