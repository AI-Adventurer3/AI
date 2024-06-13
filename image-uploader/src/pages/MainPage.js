// src/pages/MainPage.js
import React, { useState, useRef } from 'react';

function MainPage({ results, setResults, dangerousPersons }) {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [previewURLs, setPreviewURLs] = useState([]);
  const fileInputRef = useRef(null); // file input 요소에 대한 참조

  const handleFileChange = (event) => {
    const files = event.target.files;
    const newFiles = Array.from(files).filter(file => !selectedFiles.some(existingFile => existingFile.name === file.name));
    setSelectedFiles(prevFiles => [...prevFiles, ...newFiles]);
    const previews = newFiles.map((file) => URL.createObjectURL(file));
    setPreviewURLs(prevPreviews => [...prevPreviews, ...previews]);
  };

  const handleUpload = async () => {
    setLoading(true);
    const uploadPromises = selectedFiles.map(async (file) => {
      const formData = new FormData();
      formData.append('file', file);
      try {
        const response = await fetch('http://localhost:8000/classify-image/', {
          method: 'POST',
          body: formData,
        });
        const data = await response.json();
        return data;
      } catch (error) {
        console.error('이미지 업로드 에러:', error);
        return { error: error.message };
      }
    });

    try {
      const uploadResults = await Promise.all(uploadPromises);
      const updatedResults = [...results, ...uploadResults];
      setResults(updatedResults);
      checkForDangerousPersons(uploadResults);
      saveResults(updatedResults);
    } catch (error) {
      console.error('이미지 업로드 에러:', error);
    } finally {
      setLoading(false);
      setPreviewURLs([]);
      setSelectedFiles([]);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const checkForDangerousPersons = (results) => {
    const foundDangerousPersons = results.filter((result) =>
      dangerousPersons.some(
        (person) => person.name === result.name // 이름 기준으로 비교
      )
    );
    if (foundDangerousPersons.length > 0) {
      alert('경고: 위험 인물이 감지되었습니다!');
    }
  };

  const saveResults = (results) => {
    localStorage.setItem('results', JSON.stringify(results));
  };

  const handleClearResults = () => {
    localStorage.removeItem('results');
    setResults([]);
  };

  return (
    <div>
      <h1>오늘의 침입자를 알려줘</h1>
      <input type="file" onChange={handleFileChange} ref={fileInputRef} multiple />
      <div className="image-previews">
        {previewURLs.map((url, index) => (
          <img key={index} src={url} alt={`미리 보기 ${index + 1}`} style={{ maxWidth: '150px' }} />
        ))}
      </div>
      <button onClick={handleUpload} disabled={!selectedFiles.length || loading}>
        {loading ? '분석 중...' : '업로드'}
      </button>
      <button onClick={handleClearResults}>초기화</button>
      {results.map((result, index) => (
        <div key={index} className="result">
          <h2>이미지 {index + 1} 결과:</h2>
          <div key="image">
            <img src={`data:image/jpeg;base64,${result.image_base64}`} alt={`이미지 ${index + 1}`} style={{ maxWidth: '150px' }} />
            {result.captions.map((caption, idx) => (
              <div key={idx}>{caption}</div>
            ))}
            {result.emotions.map((emotion, idx) => (
              <div key={idx}>{emotion.label}: {emotion.score}</div>
            ))}
            {result.genders.map((gender, idx) => (
              <div key={idx}>{gender.label}: {gender.score}</div>
            ))}
            {result.faces.map((face, idx) => (
              <div key={idx}>{face.label}: {face.score}</div>
            ))}
            {result.objects.map((object, idx) => (
              <div key={idx}>{object.label}: {object.score}</div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

export default MainPage;
