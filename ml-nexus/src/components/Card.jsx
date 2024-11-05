// Card.js
import React from 'react';

function Card({ data }) {
  return (
    <div className="w-full max-w-xs bg-gray shadow-lg rounded-lg overflow-hidden transform transition duration-300 ease-in-out hover:scale-105 hover:shadow-xl">
      <img src={data.img} alt={data.title} className="w-full h-48 object-cover" />
      <div className="p-2">
        <h2 className="text-xl font-bold">{data.title}</h2>
        <p className="text-sm mt-2">{data.desc}</p>
        <a href={data.ref} target="_blank" rel="noopener noreferrer" className="text-green-500 hover:text-green-700 mt-4 block">Learn More</a>
      </div>
    </div>
  );
}

export default Card;
