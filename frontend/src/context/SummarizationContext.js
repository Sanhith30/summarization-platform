import React, { createContext, useContext, useReducer } from 'react';

// Initial state
const initialState = {
  currentSummary: null,
  isLoading: false,
  error: null,
  history: [],
  settings: {
    mode: 'researcher',
    useCache: true,
    maxLength: null,
    minLength: null
  }
};

// Action types
const ActionTypes = {
  SET_LOADING: 'SET_LOADING',
  SET_SUMMARY: 'SET_SUMMARY',
  SET_ERROR: 'SET_ERROR',
  CLEAR_ERROR: 'CLEAR_ERROR',
  ADD_TO_HISTORY: 'ADD_TO_HISTORY',
  UPDATE_SETTINGS: 'UPDATE_SETTINGS',
  CLEAR_SUMMARY: 'CLEAR_SUMMARY'
};

// Reducer
function summarizationReducer(state, action) {
  switch (action.type) {
    case ActionTypes.SET_LOADING:
      return {
        ...state,
        isLoading: action.payload,
        error: null
      };
    
    case ActionTypes.SET_SUMMARY:
      return {
        ...state,
        currentSummary: action.payload,
        isLoading: false,
        error: null
      };
    
    case ActionTypes.SET_ERROR:
      return {
        ...state,
        error: action.payload,
        isLoading: false
      };
    
    case ActionTypes.CLEAR_ERROR:
      return {
        ...state,
        error: null
      };
    
    case ActionTypes.ADD_TO_HISTORY:
      return {
        ...state,
        history: [action.payload, ...state.history.slice(0, 9)] // Keep last 10
      };
    
    case ActionTypes.UPDATE_SETTINGS:
      return {
        ...state,
        settings: {
          ...state.settings,
          ...action.payload
        }
      };
    
    case ActionTypes.CLEAR_SUMMARY:
      return {
        ...state,
        currentSummary: null,
        error: null
      };
    
    default:
      return state;
  }
}

// Context
const SummarizationContext = createContext();

// Provider component
export function SummarizationProvider({ children }) {
  const [state, dispatch] = useReducer(summarizationReducer, initialState);

  // Actions
  const actions = {
    setLoading: (loading) => {
      dispatch({ type: ActionTypes.SET_LOADING, payload: loading });
    },

    setSummary: (summary) => {
      dispatch({ type: ActionTypes.SET_SUMMARY, payload: summary });
      
      // Add to history
      const historyItem = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        type: summary.type || 'text',
        title: summary.title || 'Untitled',
        summary: summary.summary_short,
        processingTime: summary.processing_time
      };
      dispatch({ type: ActionTypes.ADD_TO_HISTORY, payload: historyItem });
    },

    setError: (error) => {
      dispatch({ type: ActionTypes.SET_ERROR, payload: error });
    },

    clearError: () => {
      dispatch({ type: ActionTypes.CLEAR_ERROR });
    },

    updateSettings: (settings) => {
      dispatch({ type: ActionTypes.UPDATE_SETTINGS, payload: settings });
    },

    clearSummary: () => {
      dispatch({ type: ActionTypes.CLEAR_SUMMARY });
    }
  };

  const value = {
    ...state,
    ...actions
  };

  return (
    <SummarizationContext.Provider value={value}>
      {children}
    </SummarizationContext.Provider>
  );
}

// Hook to use the context
export function useSummarization() {
  const context = useContext(SummarizationContext);
  if (!context) {
    throw new Error('useSummarization must be used within a SummarizationProvider');
  }
  return context;
}