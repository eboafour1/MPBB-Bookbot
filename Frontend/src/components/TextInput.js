import React from 'react';
import { TextField } from '@mui/material';

const TextInput = ({ text, setText }) => (
  <TextField
    label="Or paste your text here"
    multiline
    minRows={8}
    fullWidth
    variant="outlined"
    value={text}
    onChange={(e) => setText(e.target.value)}
  />
);

export default TextInput;