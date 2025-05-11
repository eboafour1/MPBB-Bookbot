import React from 'react';
import { FormControl, InputLabel, Select, MenuItem } from '@mui/material';

const ModelSelector = ({ model, setModel }) => (
  <FormControl fullWidth sx={{ mb: 2 }}>
    <InputLabel>Model</InputLabel>
    <Select value={model} label="Model" onChange={(e) => setModel(e.target.value)}>
      <MenuItem value="pegasus">PEGASUS</MenuItem>
      <MenuItem value="bart">BART</MenuItem>
      <MenuItem value="bertsum">BERTSum</MenuItem>
    </Select>
  </FormControl>
);

export default ModelSelector;