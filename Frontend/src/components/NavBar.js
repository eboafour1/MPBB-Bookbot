import React from 'react';
import { AppBar, Toolbar, Typography, IconButton } from '@mui/material';
import { Brightness4, Brightness7 } from '@mui/icons-material';

const NavBar = ({ mode, toggleMode }) => (
  <AppBar position="static">
    <Toolbar>
      <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
        Bookbot AI
      </Typography>
      <IconButton color="inherit" onClick={toggleMode}>
        {mode === 'light' ? <Brightness4 /> : <Brightness7 />} 
      </IconButton>
    </Toolbar>
  </AppBar>
);

export default NavBar;