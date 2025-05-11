import React, { useState } from "react";
import axios from "axios";

const Summarize = () => {
    const [text, setText] = useState("");
    const [summary, setSummary] = useState("");

    const summarizeText = async () => {
        try {
            const response = await axios.post("http://127.0.0.1:8000/summarize/", { text });
            setSummary(response.data.summary);
        } catch (error) {
            console.error("Error summarizing:", error);
        }
    };

    return (
        <div>
            <h1>Bookbot AI Summarizer</h1>
            <textarea value={text} onChange={(e) => setText(e.target.value)} placeholder="Enter text to summarize"/>
            <button onClick={summarizeText}>Summarize</button>
            <h2>Summary:</h2>
            <p>{summary}</p>
        </div>
    );
};

export default Summarize;
