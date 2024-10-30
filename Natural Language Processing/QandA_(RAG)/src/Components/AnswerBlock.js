import React from 'react';
import './CSS/AnswerBlock.css';

const AnswerBlock = ({ answer }) => {
  return (
    <div className="answer-block">
      <p>{answer}</p>
    </div>
  );
};

export default AnswerBlock;
