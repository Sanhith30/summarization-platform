import React, { useState } from 'react';
import InputSection from '../components/InputSection';
import SummaryDisplay from '../components/SummaryDisplay';
import SettingsPanel from '../components/SettingsPanel';
import HistoryPanel from '../components/HistoryPanel';
import { useSummarization } from '../context/SummarizationContext';
import { Settings, History, X } from 'lucide-react';

function HomePage() {
  const { currentSummary, isLoading, error } = useSummarization();
  const [showSettings, setShowSettings] = useState(false);
  const [showHistory, setShowHistory] = useState(false);

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header Actions */}
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Intelligent Summarization
          </h1>
          <p className="text-gray-600">
            Transform text, PDFs, and YouTube videos into concise, intelligent summaries
          </p>
        </div>
        
        <div className="flex space-x-3">
          <button
            onClick={() => setShowHistory(!showHistory)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
              showHistory 
                ? 'bg-primary-600 text-white' 
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            <History className="h-4 w-4" />
            <span className="hidden sm:inline">History</span>
          </button>
          
          <button
            onClick={() => setShowSettings(!showSettings)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
              showSettings 
                ? 'bg-primary-600 text-white' 
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            <Settings className="h-4 w-4" />
            <span className="hidden sm:inline">Settings</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Main Content */}
        <div className="lg:col-span-3 space-y-6">
          {/* Input Section */}
          <InputSection />
          
          {/* Error Display */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex items-start">
                <div className="flex-shrink-0">
                  <X className="h-5 w-5 text-red-400" />
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-red-800">
                    Error
                  </h3>
                  <div className="mt-2 text-sm text-red-700">
                    {error}
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {/* Loading State */}
          {isLoading && (
            <div className="card">
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <div className="loading-spinner mx-auto mb-4"></div>
                  <p className="text-gray-600">Processing your content...</p>
                  <p className="text-sm text-gray-500 mt-2">
                    This may take a few moments for large documents
                  </p>
                </div>
              </div>
            </div>
          )}
          
          {/* Summary Display */}
          {currentSummary && !isLoading && (
            <SummaryDisplay summary={currentSummary} />
          )}
          
          {/* Welcome Message */}
          {!currentSummary && !isLoading && !error && (
            <div className="card text-center py-12">
              <div className="max-w-md mx-auto">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">
                  Welcome to SummarizeAI
                </h2>
                <p className="text-gray-600 mb-6">
                  Get started by entering text, uploading a PDF, or pasting a YouTube URL above. 
                  Our AI will generate intelligent summaries with confidence scores and citations.
                </p>
                <div className="grid grid-cols-3 gap-4 text-sm text-gray-500">
                  <div>
                    <div className="font-medium">📝 Text</div>
                    <div>Any text content</div>
                  </div>
                  <div>
                    <div className="font-medium">📄 PDF</div>
                    <div>Document upload</div>
                  </div>
                  <div>
                    <div className="font-medium">🎥 YouTube</div>
                    <div>Video transcripts</div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="lg:col-span-1 space-y-6">
          {/* Settings Panel */}
          {showSettings && (
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">Settings</h3>
                <button
                  onClick={() => setShowSettings(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
              <SettingsPanel />
            </div>
          )}
          
          {/* History Panel */}
          {showHistory && (
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">Recent Summaries</h3>
                <button
                  onClick={() => setShowHistory(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
              <HistoryPanel />
            </div>
          )}
          
          {/* Features Info */}
          {!showSettings && !showHistory && (
            <div className="card">
              <h3 className="text-lg font-semibold mb-4">Features</h3>
              <div className="space-y-3 text-sm">
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    <div className="font-medium">Offline Processing</div>
                    <div className="text-gray-600">No API keys required</div>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    <div className="font-medium">Multi-Model Ensemble</div>
                    <div className="text-gray-600">BART, Pegasus, T5</div>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-purple-500 rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    <div className="font-medium">Confidence Scoring</div>
                    <div className="text-gray-600">Reliability indicators</div>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-orange-500 rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    <div className="font-medium">Query-Focused</div>
                    <div className="text-gray-600">Targeted summaries</div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default HomePage;