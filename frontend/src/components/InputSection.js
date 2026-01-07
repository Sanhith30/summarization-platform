import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { useSummarization } from '../context/SummarizationContext';
import { summarizationAPI } from '../services/api';
import { FileText, Youtube, Upload, Send, Loader } from 'lucide-react';

function InputSection() {
  const [activeTab, setActiveTab] = useState('text');
  const [textContent, setTextContent] = useState('');
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [query, setQuery] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  
  const { 
    setLoading, 
    setSummary, 
    setError, 
    clearError, 
    settings 
  } = useSummarization();

  // File drop zone
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'application/pdf': ['.pdf']
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024, // 10MB
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        setSelectedFile(acceptedFiles[0]);
        setActiveTab('pdf');
      }
    },
    onDropRejected: (rejectedFiles) => {
      const error = rejectedFiles[0]?.errors[0];
      if (error?.code === 'file-too-large') {
        setError('File too large. Maximum size is 10MB.');
      } else if (error?.code === 'file-invalid-type') {
        setError('Invalid file type. Please upload a PDF file.');
      } else {
        setError('File upload failed. Please try again.');
      }
    }
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    clearError();
    
    try {
      setLoading(true);
      let result;

      const requestData = {
        query: query.trim() || undefined,
        mode: settings.mode,
        use_cache: settings.useCache,
        ...(settings.maxLength && { max_length: settings.maxLength }),
        ...(settings.minLength && { min_length: settings.minLength })
      };

      switch (activeTab) {
        case 'text':
          if (!textContent.trim()) {
            throw new Error('Please enter some text to summarize.');
          }
          result = await summarizationAPI.summarizeText({
            content: textContent.trim(),
            ...requestData
          });
          result.type = 'text';
          result.title = `Text (${textContent.length} chars)`;
          break;

        case 'youtube':
          if (!youtubeUrl.trim()) {
            throw new Error('Please enter a YouTube URL.');
          }
          result = await summarizationAPI.summarizeYouTube({
            url: youtubeUrl.trim(),
            ...requestData
          });
          result.type = 'youtube';
          result.title = result.video_info?.title || 'YouTube Video';
          break;

        case 'pdf':
          if (!selectedFile) {
            throw new Error('Please select a PDF file.');
          }
          result = await summarizationAPI.summarizePDF(selectedFile, requestData);
          result.type = 'pdf';
          result.title = result.document_info?.filename || selectedFile.name;
          break;

        default:
          throw new Error('Invalid input type.');
      }

      setSummary(result);
      
      // Clear form
      setTextContent('');
      setYoutubeUrl('');
      setQuery('');
      setSelectedFile(null);
      
    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const isFormValid = () => {
    switch (activeTab) {
      case 'text':
        return textContent.trim().length > 0;
      case 'youtube':
        return youtubeUrl.trim().length > 0;
      case 'pdf':
        return selectedFile !== null;
      default:
        return false;
    }
  };

  return (
    <div className="card">
      {/* Tab Navigation */}
      <div className="flex space-x-1 mb-6 bg-gray-100 p-1 rounded-lg">
        <button
          onClick={() => setActiveTab('text')}
          className={`flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            activeTab === 'text'
              ? 'bg-white text-primary-600 shadow-sm'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          <FileText className="h-4 w-4" />
          <span>Text</span>
        </button>
        
        <button
          onClick={() => setActiveTab('youtube')}
          className={`flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            activeTab === 'youtube'
              ? 'bg-white text-primary-600 shadow-sm'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          <Youtube className="h-4 w-4" />
          <span>YouTube</span>
        </button>
        
        <button
          onClick={() => setActiveTab('pdf')}
          className={`flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            activeTab === 'pdf'
              ? 'bg-white text-primary-600 shadow-sm'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          <Upload className="h-4 w-4" />
          <span>PDF</span>
        </button>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Content Input */}
        {activeTab === 'text' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Text Content
            </label>
            <textarea
              value={textContent}
              onChange={(e) => setTextContent(e.target.value)}
              placeholder="Paste your text here... (minimum 10 characters)"
              className="input-field h-32 resize-none"
              maxLength={1000000}
            />
            <div className="text-xs text-gray-500 mt-1">
              {textContent.length.toLocaleString()} characters
            </div>
          </div>
        )}

        {activeTab === 'youtube' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              YouTube URL
            </label>
            <input
              type="url"
              value={youtubeUrl}
              onChange={(e) => setYoutubeUrl(e.target.value)}
              placeholder="https://youtube.com/watch?v=..."
              className="input-field"
            />
            <div className="text-xs text-gray-500 mt-1">
              Supports youtube.com and youtu.be URLs
            </div>
          </div>
        )}

        {activeTab === 'pdf' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              PDF Document
            </label>
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
                isDragActive
                  ? 'border-primary-500 bg-primary-50'
                  : selectedFile
                  ? 'border-green-500 bg-green-50'
                  : 'border-gray-300 hover:border-gray-400'
              }`}
            >
              <input {...getInputProps()} />
              {selectedFile ? (
                <div className="space-y-2">
                  <Upload className="h-8 w-8 text-green-500 mx-auto" />
                  <div className="text-sm font-medium text-green-700">
                    {selectedFile.name}
                  </div>
                  <div className="text-xs text-green-600">
                    {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                  </div>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedFile(null);
                    }}
                    className="text-xs text-red-600 hover:text-red-800"
                  >
                    Remove file
                  </button>
                </div>
              ) : (
                <div className="space-y-2">
                  <Upload className="h-8 w-8 text-gray-400 mx-auto" />
                  <div className="text-sm text-gray-600">
                    {isDragActive
                      ? 'Drop the PDF file here...'
                      : 'Drag & drop a PDF file here, or click to select'}
                  </div>
                  <div className="text-xs text-gray-500">
                    Maximum file size: 10MB
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Query Input */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Focus Query (Optional)
          </label>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="What specific aspect would you like to focus on?"
            className="input-field"
            maxLength={500}
          />
          <div className="text-xs text-gray-500 mt-1">
            Add a query to get targeted summaries focused on specific topics
          </div>
        </div>

        {/* Submit Button */}
        <div className="flex justify-end">
          <button
            type="submit"
            disabled={!isFormValid()}
            className="btn-primary flex items-center space-x-2"
          >
            <Send className="h-4 w-4" />
            <span>Generate Summary</span>
          </button>
        </div>
      </form>
    </div>
  );
}

export default InputSection;