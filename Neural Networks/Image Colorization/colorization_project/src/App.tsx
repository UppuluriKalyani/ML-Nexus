import React, { useState } from 'react';
import ImageUpload from './components/ImageUpload';
import ImageDisplay from './components/ImageDisplay';
import { Palette } from 'lucide-react';

const App: React.FC = () => {
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [colorizedImage, setColorizedImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageUpload = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      setOriginalImage(e.target?.result as string);
      setColorizedImage(null);
      setError(null);
    };
    reader.readAsDataURL(file);
  };

  const handleColorize = async () => {
    if (!originalImage) return;
    setIsProcessing(true);
    setError(null);
    try {
      const base64Data = originalImage.split(',')[1];
      
      const response = await fetch('https://8f54-34-143-175-163.ngrok-free.app/colorize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: base64Data }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      if (data.error) {
        throw new Error(data.error);
      }
      
      setColorizedImage(data.colorizedImageUrl);
    } catch (error) {
      console.error('Error colorizing image:', error);
      setError(`Failed to colorize image: ${error.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 py-6 flex flex-col justify-center sm:py-12 relative overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0 z-0">
        <div className="absolute inset-0 bg-gradient-to-r from-purple-400 via-pink-500 to-red-500 opacity-30"></div>
        <div className="absolute inset-0 animate-pulse bg-gradient-to-l from-yellow-400 via-red-500 to-pink-500 opacity-20"></div>
        <div className="absolute inset-0 animate-pulse delay-1000 bg-gradient-to-t from-blue-400 via-green-500 to-yellow-500 opacity-20"></div>
      </div>

      <div className="relative py-3 sm:max-w-xl sm:mx-auto z-10">
        <div className="absolute inset-0 bg-gradient-to-r from-cyan-400 to-light-blue-500 shadow-lg transform -skew-y-6 sm:skew-y-0 sm:-rotate-6 sm:rounded-3xl"></div>
        <div className="relative px-4 py-10 bg-white shadow-lg sm:rounded-3xl sm:p-20">
          <div className="max-w-md mx-auto">
            <div className="flex items-center space-x-5">
              <Palette className="h-14 w-14 text-blue-500" />
              <div className="text-2xl font-bold">Image Colorizer</div>
            </div>
            <div className="divide-y divide-gray-200">
              <div className="py-8 text-base leading-6 space-y-4 text-gray-700 sm:text-lg sm:leading-7">
                <ImageUpload onImageUpload={handleImageUpload} />
                <div className="flex justify-center">
                  <button
                    onClick={handleColorize}
                    disabled={!originalImage || isProcessing}
                    className="px-4 py-2 font-bold text-white bg-blue-500 rounded-full hover:bg-blue-700 focus:outline-none focus:shadow-outline disabled:opacity-50 transition duration-300 ease-in-out transform hover:-translate-y-1 hover:scale-110"
                  >
                    {isProcessing ? 'Processing...' : 'Colorize Image'}
                  </button>
                </div>
                {error && (
                  <div className="text-red-500 text-center">{error}</div>
                )}
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <ImageDisplay title="Original Image" imageUrl={originalImage} />
                  <ImageDisplay title="Colorized Image" imageUrl={colorizedImage} />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;