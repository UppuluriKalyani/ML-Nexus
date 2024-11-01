import React from 'react';
import Btn from './Btn';

const ContributeSection = () => {
  return (
    <section className="py-16 px-4 bg-[#EDF7F6] dark:bg-[#253341] text-[#28333F] dark:text-[#AFC2CB]">
      <div className="max-w-3xl mx-auto bg-white dark:bg-[#324655] rounded-lg p-8 text-center shadow-md">
        <h2 className="text-2xl md:text-3xl font-semibold mb-4">
          Each and Every Contribution Matters
        </h2>
        <p className="text-base md:text-lg mb-8">
          Join us in building something great. Your unique skills and knowledge can make a difference!
        </p>
        <Btn className='mx-auto' value="Contribute Now" /> 
      </div>
    </section>
  );
};

export default ContributeSection;
