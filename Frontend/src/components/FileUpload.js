import React from 'react';
import { Button } from '@mui/material';
import * as pdfjsLib from 'pdfjs-dist/legacy/build/pdf';
import ePub from 'epubjs';
import mammoth from 'mammoth';

pdfjsLib.GlobalWorkerOptions.workerSrc =
  `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.js`;

const FileUpload = ({ onFileRead }) => {
  const handleChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const ext = file.name.split('.').pop().toLowerCase();
    let text = '';
    try {
      if (ext === 'pdf') {
        const arrayBuffer = await file.arrayBuffer();
        const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
        for (let i = 1; i <= pdf.numPages; i++) {
          const page = await pdf.getPage(i);
          const content = await page.getTextContent();
          text += content.items.map(item => item.str).join(' ') + ' ';
        }
      } else if (ext === 'txt') {
        text = await file.text();
      } else if (ext === 'epub') {
        const arrayBuffer = await file.arrayBuffer();
        const book = ePub(arrayBuffer);
        const { spine } = book;
        for (const item of spine.spineItems) {
          const html = await item.load(book.load.bind(book)).then(() => item.render());
          text += html.replace(/<[^>]+>/g, ' ') + ' ';
        }
      } else if (ext === 'docx') {
        const arrayBuffer = await file.arrayBuffer();
        const result = await mammoth.extractRawText({ arrayBuffer });
        text = result.value;
      } else if (ext === 'mobi') {
        alert('MOBI format not supported. Please convert to EPUB or PDF first.');
        return;
      } else {
        alert('Unsupported file format.');
        return;
      }
      onFileRead(text);
    } catch (err) {
      console.error('File parsing error:', err);
      alert('Error reading file. Please try another file.');
    }
  };

  return (
    <Button variant="outlined" component="label" fullWidth>
      Upload File (.txt, .pdf, .epub, .docx)
      <input hidden type="file" accept=".txt,.pdf,.epub,.docx,.mobi" onChange={handleChange} />
    </Button>
  );
};

export default FileUpload;
