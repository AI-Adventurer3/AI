import React, { useRef } from 'react';

function RegisterPage({ dangerousPersons, setDangerousPersons }) {
  const fileInputRef = useRef(null);

  const handleRegister = (event) => {
    const files = event.target.files;
    if (files.length === 0) return;

    const newPerson = {
      id: new Date().getTime(), // 각 인물의 고유 ID
      name: '이름 없음', // 기본 이름
      url: URL.createObjectURL(files[0]),
    };

    const updatedDangerousPersons = [...dangerousPersons, newPerson];
    setDangerousPersons(updatedDangerousPersons);
    localStorage.setItem('dangerousPersons', JSON.stringify(updatedDangerousPersons));

    // 파일 선택 input 초기화
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const updateName = (id, newName) => {
    const updatedDangerousPersons = dangerousPersons.map((person) =>
      person.id === id ? { ...person, name: newName } : person
    );
    setDangerousPersons(updatedDangerousPersons);
    localStorage.setItem('dangerousPersons', JSON.stringify(updatedDangerousPersons));
  };

  const deletePerson = (id) => {
    const updatedDangerousPersons = dangerousPersons.filter((person) => person.id !== id);
    setDangerousPersons(updatedDangerousPersons);
    localStorage.setItem('dangerousPersons', JSON.stringify(updatedDangerousPersons));
  };

  const handleClearAll = () => {
    setDangerousPersons([]);
    localStorage.removeItem('dangerousPersons');
  };

  return (
    <div>
      <h2>위험인물 등록 페이지</h2>
      <input
        type="file"
        onChange={handleRegister}
        accept="image/*"
        ref={fileInputRef}
      />
      <h3>등록된 위험인물</h3>
      <ul>
        {dangerousPersons.map((person) => (
          <li key={person.id}>
            <img src={person.url} alt={person.name} style={{ maxWidth: '100px' }} />
            <input
              type="text"
              value={person.name}
              onChange={(e) => updateName(person.id, e.target.value)}
              placeholder="이름 수정"
            />
            <button onClick={() => deletePerson(person.id)}>삭제</button>
          </li>
        ))}
      </ul>
      <button onClick={handleClearAll}>
        모든 위험인물 초기화
      </button>
    </div>
  );
}

export default RegisterPage;
