import React from 'react';
import { ToggleButtonGroup, ToggleButton } from '@mui/material';

const SummaryLengthSelector = ({ length, setLength }) => (
  <ToggleButtonGroup
    value={length}
    exclusive
    onChange={(_, val) => val && setLength(val)}
    sx={{ mb: 2 }}
  >
    <ToggleButton value="short">Short</ToggleButton>
    <ToggleButton value="medium">Medium</ToggleButton>
    <ToggleButton value="long">Long</ToggleButton>
  </ToggleButtonGroup>
);

export default SummaryLengthSelector;
