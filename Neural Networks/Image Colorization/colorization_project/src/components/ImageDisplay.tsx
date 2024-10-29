import React from 'react';

interface ImageDisplayProps {
  title: string;
  imageUrl: string | null;
}

const ImageDisplay: React.FC<ImageDisplayProps> = ({ title, imageUrl }) => {
  return (
    <div>
      <h2 className="text-lg font-medium text-gray-900 mb-2">{title}</h2>
      {imageUrl ? (
        <img src={imageUrl} alt={title} className="w-full h-64 object-cover rounded-lg" />
      ) : (
        <div className="w-full h-64 bg-gray-200 flex items-center justify-center rounded-lg">
          <p className="text-gray-500">No image uploaded</p>
        </div>
      )}
    </div>
  );
};

export default ImageDisplay;