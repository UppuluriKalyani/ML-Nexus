// import React, { useState } from 'react';
// import axios from 'axios';
// import { useNavigate } from 'react-router-dom';

// const FileUpload = ({ setSummary }) => {
//     const [file, setFile] = useState(null);
//     const [message, setMessage] = useState('');
//     const navigate = useNavigate();

//     const handleFileChange = (event) => {
//         setFile(event.target.files[0]);
//     };

//     const handleUpload = async (event) => {
//         event.preventDefault();

//         if (!file) {
//             setMessage('No file selected for uploading');
//             return;
//         }

//         const formData = new FormData();
//         formData.append('file', file);

//         try {
//             const response = await axios.post('http://localhost:5000/upload', formData, {
//                 headers: {
//                     'Content-Type': 'multipart/form-data',
//                 },
//             });

//             setMessage(response.data.message);
//             setSummary(response.data.summary);
//             navigate('/summary');
//         } catch (error) {
//             if (error.response) {
//                 setMessage(error.response.data.error);
//             } else {
//                 setMessage('An error occurred while uploading the file.');
//             }
//         }
//     };

//     return (
//         <div>
//             <h1>Upload a File</h1>
//             <form onSubmit={handleUpload} encType="multipart/form-data">
//                 <input type="file" name="file" onChange={handleFileChange} accept=".pdf,.txt" />
//                 <button type="submit" className="button2 blue radius-16">Upload a PDF or TXT</button>
//             </form>
//             {message && <p>{message}</p>}
//         </div>
//     );
// };

// export default FileUpload;

import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const FileUpload = () => {
    const [file, setFile] = useState(null);
    const [message, setMessage] = useState('');
    const navigate = useNavigate();

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
    };

    const handleUpload = async (event) => {
        event.preventDefault();

        if (!file) {
            setMessage('No file selected for uploading');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post('http://localhost:5000/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            setMessage(response.data.message);
            navigate('/summary', { state: { summary: response.data.summary } });
        } catch (error) {
            if (error.response) {
                setMessage(error.response.data.error);
            } else {
                setMessage('An error occurred while uploading the file.');
            }
        }
    };

    return (
        <div>
            <h1>Upload a File</h1>
            <form onSubmit={handleUpload} encType="multipart/form-data">
                <input type="file" name="file" onChange={handleFileChange} accept=".pdf,.txt" />
                <button type="submit" className="button2 blue radius-16">Upload a PDF or TXT</button>
            </form>
            {message && <p>{message}</p>}
        </div>
    );
};

export default FileUpload;
