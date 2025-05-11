// File: src/index.js
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import getTheme from './theme';
import { ThemeProvider, CssBaseline } from '@mui/material';

// 1️⃣ Grab your root container
const container = document.getElementById('root');
// 2️⃣ Create the new React 18 root
const root = ReactDOM.createRoot(container);

// 3️⃣ Render your app inside ThemeProvider
root.render(
  <React.StrictMode>
    <ThemeProvider theme={getTheme('light')}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  </React.StrictMode>
);
