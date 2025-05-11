import { createTheme } from '@mui/material/styles';

const getTheme = (mode) =>
  createTheme({
    palette: {
      mode,
      primary: { main: '#1976d2' },
      secondary: { main: '#f50057' }
    },
    typography: {
      h3: { fontWeight: 700, marginBottom: '1rem' },
      h6: { color: '#555' }
    }
  });

export default getTheme;