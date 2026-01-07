import React from 'react';
import { useSummarization } from '../context/SummarizationContext';
import { FileText, Youtube, Upload, Clock, Trash2 } from 'lucide-react';

function HistoryPanel() {
  const { history, setSummary } = useSummarization();

  const getTypeIcon = (type) => {
    switch (type) {
      case 'youtube':
        return <Youtube className="h-4 w-4 text-red-500" />;
      case 'pdf':
        return <Upload className="h-4 w-4 text-blue-500" />;
      default:
        return <FileText className="h-4 w-4 text-gray-500" />;
    }
  };

  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    
    return date.toLocaleDateString();
  };

  const truncateText = (text, maxLength = 100) => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  if (history.length === 0) {
    return (
      <div className="text-center py-8">
        <div className="text-gray-400 mb-2">
          <Clock className="h-8 w-8 mx-auto" />
        </div>
        <p className="text-sm text-gray-500">No summaries yet</p>
        <p className="text-xs text-gray-400 mt-1">
          Your recent summaries will appear here
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {history.map((item) => (
        <div
          key={item.id}
          className="summary-card cursor-pointer"
          onClick={() => {
            // In a real app, you'd fetch the full summary data
            // For now, we'll just show what we have
            setSummary({
              ...item,
              summary_short: item.summary,
              summary_medium: item.summary,
              summary_detailed: item.summary,
              key_points: [],
              query_focused_summary: '',
              timestamps: [],
              confidence_scores: [0.8],
              citations: [],
              explainability: {
                method: 'cached',
                confidence: 0.8,
                source_alignment: 'high',
                chunk_count: 1,
                models_used: 1,
                processing_stages: ['cached_retrieval']
              }
            });
          }}
        >
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0 mt-1">
              {getTypeIcon(item.type)}
            </div>
            
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between mb-1">
                <h4 className="text-sm font-medium text-gray-900 truncate">
                  {item.title}
                </h4>
                <span className="text-xs text-gray-500 flex-shrink-0 ml-2">
                  {formatTime(item.timestamp)}
                </span>
              </div>
              
              <p className="text-xs text-gray-600 mb-2">
                {truncateText(item.summary)}
              </p>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2 text-xs text-gray-500">
                  <Clock className="h-3 w-3" />
                  <span>{item.processingTime?.toFixed(1)}s</span>
                </div>
                
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    // In a real app, you'd remove from history
                    console.log('Remove item:', item.id);
                  }}
                  className="text-gray-400 hover:text-red-500 transition-colors"
                  title="Remove from history"
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              </div>
            </div>
          </div>
        </div>
      ))}
      
      {history.length >= 10 && (
        <div className="text-center pt-2">
          <button className="text-xs text-gray-500 hover:text-gray-700">
            Clear old history
          </button>
        </div>
      )}
    </div>
  );
}

export default HistoryPanel;