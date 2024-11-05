// About.jsx
import React from 'react';

function About() {
  return (
    <div className="h-full w-full flex flex-col items-center p-6 lg:p-12 bg-gray-50">
      <h1 className="text-3xl lg:text-5xl font-bold text-center text-gray-800 mb-8">About Us</h1>

      <div className="max-w-4xl mx-auto text-center">
        <p className="text-lg text-gray-600 mb-4">
          Welcome to <span className="text-green-500">ML Nexus</span>, a collaborative platform for machine learning enthusiasts.
          Our goal is to provide cutting-edge tools, resources, and research in machine learning to accelerate your learning and project development.
        </p>
        <p className="text-lg text-gray-600 mb-4">
          Whether you're new to machine learning or a seasoned professional, you'll find a vibrant community of developers,
          researchers, and contributors who share knowledge, collaborate on projects, and push the boundaries of what's possible.
        </p>
        <p className="text-lg text-gray-600 mb-4">
          Our mission is to empower individuals and teams to build and deploy machine learning models, while fostering an open and inclusive community.
        </p>
      </div>

      <div className="mt-12">
        <h2 className="text-2xl font-semibold text-center text-gray-800">Our Team</h2>
        <div className="flex flex-wrap justify-center mt-8">
          {/* Add team member info */}
          <div className="max-w-xs mx-4 mb-6">
            <div className="bg-white rounded-lg shadow-md p-6">
              <img src="https://via.placeholder.com/150" alt="Team Member" className="w-32 h-32 mx-auto rounded-full mb-4" />
              <h3 className="text-xl font-semibold text-gray-800">John Doe</h3>
              <p className="text-gray-600">Machine Learning Engineer</p>
            </div>
          </div>
          {/* Add more team members as needed */}
        </div>
      </div>
    </div>
  );
}

export default About;
