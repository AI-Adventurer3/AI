import React, { useState } from 'react';

const ImageUpload = ({ onAddImage }) => {
    const [preview, setPreview] = useState(null);
    const [file, setFile] = useState(null);

    const handleImageChange = (event) => {
        const selectedFile = event.target.files[0];
        if (selectedFile) {
            setFile(selectedFile);
            const reader = new FileReader();
            reader.onloadend = () => {
                setPreview(reader.result);
            };
            reader.readAsDataURL(selectedFile);
        }
    };

    const handleUpload = () => {
        if (!file) return;
        const newImage = {
            id: new Date().getTime(),
            src: preview,
            name: file.name,
        };
        onAddImage(newImage);
        setFile(null);
        setPreview(null);
    };

    return (
        <div className="image-upload">
            <input 
                id="file-upload" 
                type="file" 
                accept="image/*" 
                onChange={handleImageChange} 
                style={{ display: 'none' }} 
            />
            <label htmlFor="file-upload" className="custom-file-upload">
                파일 선택
            </label>
            {preview && (
                <div className="image-preview">
                    <h2>미리보기</h2>
                    <img src={preview} alt="미리보기" />
                    <button onClick={handleUpload} className="upload-button">업로드</button>
                </div>
            )}
        </div>
    );
};

export default ImageUpload;