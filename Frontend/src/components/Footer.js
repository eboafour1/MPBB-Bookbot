import React from 'react';
import { Box, Typography } from '@mui/material';

const Footer = () => (
  <Box component="footer" sx={{ p: 2, textAlign: 'center', bgcolor: 'background.paper' }}>
    <Typography variant="body2" color="text.secondary">
      © {new Date().getFullYear()} Bookbot AI. All rights reserved.
    </Typography>
  </Box>
);

export default Footer;