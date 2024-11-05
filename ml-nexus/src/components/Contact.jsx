// Contact.jsx
import React, { useState } from 'react';

function Contact() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');
  const [status, setStatus] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    // Here you can implement your form submission logic (e.g., sending data to a backend)
    setStatus('Thank you for your message! We will get back to you shortly.');
    // Clear the form after submission
    setName('');
    setEmail('');
    setMessage('');
  };

  return (
    <div className="h-full w-full flex flex-col items-center p-6 lg:p-12 bg-gray-50">
      <h1 className="text-3xl lg:text-5xl font-bold text-center text-gray-800 mb-8">Contact Us</h1>

      <div className="max-w-2xl mx-auto bg-white p-8 shadow-lg rounded-lg">
        <h2 className="text-2xl font-semibold text-center text-gray-800 mb-6">We'd Love to Hear From You!</h2>

        {status && <p className="text-center text-green-500 mb-4">{status}</p>}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="name" className="block text-gray-700 font-semibold">Full Name</label>
            <input
              type="text"
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full px-4 py-2 mt-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
              required
            />
          </div>

          <div>
            <label htmlFor="email" className="block text-gray-700 font-semibold">Email Address</label>
            <input
              type="email"
              id="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-4 py-2 mt-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
              required
            />
          </div>

          <div>
            <label htmlFor="message" className="block text-gray-700 font-semibold">Your Message</label>
            <textarea
              id="message"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              className="w-full px-4 py-2 mt-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
              rows="6"
              required
            />
          </div>

          <div className="text-center">
            <button type="submit" className="px-6 py-3 bg-green-500 text-white font-semibold rounded-md hover:bg-green-400 transition-all duration-300">
              Send Message
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default Contact;
