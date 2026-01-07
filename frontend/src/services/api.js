import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1',
  timeout: 300000, // 5 minutes for long processing
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add timestamp to prevent caching
    config.params = {
      ...config.params,
      _t: Date.now()
    };
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Handle common errors
    if (error.response) {
      // Server responded with error status
      const message = error.response.data?.detail || error.response.data?.error || 'An error occurred';
      throw new Error(message);
    } else if (error.request) {
      // Request was made but no response received
      throw new Error('No response from server. Please check your connection.');
    } else {
      // Something else happened
      throw new Error(error.message || 'An unexpected error occurred');
    }
  }
);

// API methods
export const summarizationAPI = {
  // Text summarization
  summarizeText: async (data) => {
    const response = await api.post('/summarize/text', data);
    return response.data;
  },

  // YouTube summarization
  summarizeYouTube: async (data) => {
    const response = await api.post('/summarize/youtube', data);
    return response.data;
  },

  // PDF summarization
  summarizePDF: async (file, options = {}) => {
    const formData = new FormData();
    formData.append('file', file);
    
    // Add optional parameters
    if (options.query) formData.append('query', options.query);
    if (options.mode) formData.append('mode', options.mode);
    if (options.use_cache !== undefined) formData.append('use_cache', options.use_cache);

    const response = await api.post('/summarize/pdf', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Stream text summarization
  streamTextSummarization: async (data, onProgress) => {
    const response = await fetch(`${api.defaults.baseURL}/summarize/text/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    try {
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            
            if (data === '[DONE]') {
              return;
            }

            try {
              const parsed = JSON.parse(data);
              onProgress(parsed);
            } catch (e) {
              // Ignore parsing errors for incomplete chunks
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  },

  // Submit feedback
  submitFeedback: async (data) => {
    const response = await api.post('/feedback', data);
    return response.data;
  },

  // Health check
  healthCheck: async () => {
    const response = await api.get('/health');
    return response.data;
  },

  // Get cache stats
  getCacheStats: async () => {
    const response = await api.get('/health/cache');
    return response.data;
  },

  // Get model stats
  getModelStats: async () => {
    const response = await api.get('/health/models');
    return response.data;
  }
};

export default api;