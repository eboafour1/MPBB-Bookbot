import React from 'react';
import { Button } from '@mui/material';
import { getDocument } from 'pdfjs-dist';
import * as epubjs from 'epubjs';
import mammoth from 'mammoth';

export default function FileUpload({ onText }) {
  const handleChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const ext = file.name.split('.').pop().toLowerCase();

    let text = '';
    try {
      if (ext === 'txt') {
        text = await file.text();
      }
      else if (ext === 'pdf') {
        // PDF parsing via pdfjs
        const arrayBuffer = await file.arrayBuffer();
        const pdf = await getDocument({ data: arrayBuffer }).promise;
        let full = '';
        for (let i = 1; i <= pdf.numPages; i++) {
          const page = await pdf.getPage(i);
          const content = await page.getTextContent();
          full += content.items.map(item => item.str).join(' ') + '\n\n';
        }
        text = full;
      }
      else if (ext === 'epub') {
        // EPUB parsing via epubjs
        const book = epubjs(file);
        await book.ready;
        const n = book.spine.length;
        let full = '';
        for (let i = 0; i < n; i++) {
          const chapter = await book.spine.get(i).load(book.load.bind(book));
          full += chapter.text() + '\n\n';
        }
        text = full;
      }
      else if (ext === 'docx') {
        // DOCX parsing via mammoth
        const arrayBuffer = await file.arrayBuffer();
        const result = await mammoth.extractRawText({ arrayBuffer });
        text = result.value;
      }
      else {
        // For MOBI or other formats, hand off to backend
        onText(null, file);
        return;
      }

      // deliver the plain text back up
      onText(text, null);
    } catch (err) {
      console.error('Front-end conversion error:', err);
      // fallback to backend
      onText(null, file);
    }
  };

  return (
    <Button variant="outlined" component="label" fullWidth>
      Upload File
      <input hidden type="file" onChange={handleChange} />
    </Button>
  );
}