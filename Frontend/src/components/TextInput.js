import React from 'react';
import { TextField } from '@mui/material';

export default function TextInput({ text, setText, rows = 8 }) {
  return (
    <TextField
      label="Text to summarize"
      multiline
      rows={rows}
      value={text}
      onChange={e => setText(e.target.value)}
      fullWidth
    />
  );
}
