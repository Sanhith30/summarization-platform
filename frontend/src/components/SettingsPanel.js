import React from 'react';
import { useSummarization } from '../context/SummarizationContext';

function SettingsPanel() {
  const { settings, updateSettings } = useSummarization();

  const handleModeChange = (mode) => {
    updateSettings({ mode });
  };

  const handleCacheToggle = () => {
    updateSettings({ useCache: !settings.useCache });
  };

  const handleLengthChange = (type, value) => {
    const numValue = value ? parseInt(value) : null;
    updateSettings({ [type]: numValue });
  };

  const modes = [
    {
      id: 'student',
      name: 'Student',
      description: 'Simplified explanations and key concepts'
    },
    {
      id: 'researcher',
      name: 'Researcher',
      description: 'Detailed analysis with technical depth'
    },
    {
      id: 'business',
      name: 'Business',
      description: 'Executive summaries and actionable insights'
    },
    {
      id: 'beginner',
      name: 'Beginner',
      description: 'Basic explanations and context'
    },
    {
      id: 'expert',
      name: 'Expert',
      description: 'Comprehensive technical summaries'
    }
  ];

  return (
    <div className="space-y-6">
      {/* User Mode */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-3">
          Summarization Mode
        </label>
        <div className="space-y-2">
          {modes.map((mode) => (
            <label key={mode.id} className="flex items-start space-x-3 cursor-pointer">
              <input
                type="radio"
                name="mode"
                value={mode.id}
                checked={settings.mode === mode.id}
                onChange={() => handleModeChange(mode.id)}
                className="mt-1 h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300"
              />
              <div className="flex-1">
                <div className="text-sm font-medium text-gray-900">
                  {mode.name}
                </div>
                <div className="text-xs text-gray-500">
                  {mode.description}
                </div>
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* Summary Length Controls */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-3">
          Summary Length (Optional)
        </label>
        <div className="space-y-3">
          <div>
            <label className="block text-xs text-gray-600 mb-1">
              Minimum Length (tokens)
            </label>
            <input
              type="number"
              min="10"
              max="200"
              value={settings.minLength || ''}
              onChange={(e) => handleLengthChange('minLength', e.target.value)}
              placeholder="Auto"
              className="input-field text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-600 mb-1">
              Maximum Length (tokens)
            </label>
            <input
              type="number"
              min="20"
              max="500"
              value={settings.maxLength || ''}
              onChange={(e) => handleLengthChange('maxLength', e.target.value)}
              placeholder="Auto"
              className="input-field text-sm"
            />
          </div>
        </div>
        <div className="text-xs text-gray-500 mt-2">
          Leave empty for automatic length based on content and mode
        </div>
      </div>

      {/* Cache Settings */}
      <div>
        <label className="flex items-center space-x-3 cursor-pointer">
          <input
            type="checkbox"
            checked={settings.useCache}
            onChange={handleCacheToggle}
            className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
          />
          <div className="flex-1">
            <div className="text-sm font-medium text-gray-900">
              Enable Caching
            </div>
            <div className="text-xs text-gray-500">
              Cache results for faster repeated requests
            </div>
          </div>
        </label>
      </div>

      {/* Performance Info */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="text-sm font-medium text-gray-900 mb-2">
          Performance Tips
        </h4>
        <ul className="text-xs text-gray-600 space-y-1">
          <li>• Caching improves speed for repeated content</li>
          <li>• Shorter length limits reduce processing time</li>
          <li>• Expert mode provides most detailed analysis</li>
          <li>• Query focus improves relevance</li>
        </ul>
      </div>

      {/* Model Info */}
      <div className="bg-blue-50 rounded-lg p-4">
        <h4 className="text-sm font-medium text-blue-900 mb-2">
          AI Models Used
        </h4>
        <div className="text-xs text-blue-700 space-y-1">
          <div>• BART: News and article summarization</div>
          <div>• Pegasus: Long document processing</div>
          <div>• T5: General text-to-text tasks</div>
        </div>
        <div className="text-xs text-blue-600 mt-2">
          All models run offline - no data sent to external APIs
        </div>
      </div>
    </div>
  );
}

export default SettingsPanel;