// File: src/components/SummaryLengthSelector.js
import React from 'react';
import { ToggleButtonGroup, ToggleButton } from '@mui/material';

export default function SummaryLengthSelector({ length, setLength, options }) {
  return (
    <ToggleButtonGroup
      value={length}
      exclusive
      onChange={(_, val) => val && setLength(val)}
      sx={{ mb: 2 }}
    >
      {options.map(opt => (
        <ToggleButton key={opt.value} value={opt.value}>
          {opt.label}
        </ToggleButton>
      ))}
    </ToggleButtonGroup>
  );
}
