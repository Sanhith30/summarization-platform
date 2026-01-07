import React from 'react';
import { 
  Brain, 
  Zap, 
  Shield, 
  Target, 
  Cpu, 
  Database,
  CheckCircle,
  Github,
  ExternalLink
} from 'lucide-react';

function AboutPage() {
  const features = [
    {
      icon: <Brain className="h-6 w-6 text-blue-500" />,
      title: "Multi-Model AI Ensemble",
      description: "Combines BART, Pegasus, and T5 models for superior summarization quality with confidence scoring."
    },
    {
      icon: <Shield className="h-6 w-6 text-green-500" />,
      title: "Offline & Private",
      description: "No API keys required. All processing happens locally, ensuring your data never leaves your infrastructure."
    },
    {
      icon: <Target className="h-6 w-6 text-purple-500" />,
      title: "Query-Focused Summaries",
      description: "Get targeted summaries focused on specific topics or questions you care about most."
    },
    {
      icon: <Zap className="h-6 w-6 text-yellow-500" />,
      title: "Real-Time Processing",
      description: "Hierarchical map-reduce approach with streaming responses for fast, scalable summarization."
    },
    {
      icon: <Cpu className="h-6 w-6 text-red-500" />,
      title: "Intelligent Chunking",
      description: "Adaptive token-aware segmentation with topic boundary detection for optimal processing."
    },
    {
      icon: <Database className="h-6 w-6 text-indigo-500" />,
      title: "Smart Caching",
      description: "Hash-based caching system reduces processing time for repeated content while maintaining accuracy."
    }
  ];

  const capabilities = [
    "Text documents of any length",
    "PDF files with structure detection",
    "YouTube videos with transcript extraction",
    "Multiple summary lengths (short, medium, detailed)",
    "Confidence scoring and explainability",
    "Citation mapping and source alignment",
    "Adaptive user modes (student, researcher, business, etc.)",
    "Timestamp-aware video summaries",
    "Hallucination detection and control"
  ];

  const architecture = [
    {
      step: "Input Processing",
      description: "Multi-format content ingestion with validation and preprocessing"
    },
    {
      step: "Adaptive Chunking",
      description: "Token-aware segmentation with topic boundary detection"
    },
    {
      step: "Relevance Scoring",
      description: "Query-focused filtering and content prioritization"
    },
    {
      step: "Parallel Summarization",
      description: "Map-reduce approach with model ensemble processing"
    },
    {
      step: "Hierarchical Reduction",
      description: "Multi-level summary merging and refinement"
    },
    {
      step: "Validation & Scoring",
      description: "Confidence calculation and fact-consistency checking"
    }
  ];

  return (
    <div className="max-w-4xl mx-auto space-y-12">
      {/* Hero Section */}
      <div className="text-center">
        <div className="bg-primary-600 p-4 rounded-full w-20 h-20 mx-auto mb-6 flex items-center justify-center">
          <Brain className="h-10 w-10 text-white" />
        </div>
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          SummarizeAI Platform
        </h1>
        <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
          Production-ready, offline-first intelligent summarization platform that transforms 
          text, PDFs, and YouTube videos into accurate, explainable summaries using 
          state-of-the-art AI models.
        </p>
        <div className="flex justify-center space-x-4">
          <a
            href="https://github.com/your-repo/summarization-platform"
            target="_blank"
            rel="noopener noreferrer"
            className="btn-primary flex items-center space-x-2"
          >
            <Github className="h-5 w-5" />
            <span>View on GitHub</span>
          </a>
          <a
            href="/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="btn-secondary flex items-center space-x-2"
          >
            <ExternalLink className="h-5 w-5" />
            <span>API Documentation</span>
          </a>
        </div>
      </div>

      {/* Key Features */}
      <section>
        <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center">
          Key Features
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <div key={index} className="card">
              <div className="flex items-start space-x-4">
                <div className="flex-shrink-0">
                  {feature.icon}
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-gray-600 text-sm">
                    {feature.description}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Architecture */}
      <section>
        <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center">
          How It Works
        </h2>
        <div className="card">
          <div className="space-y-6">
            {architecture.map((step, index) => (
              <div key={index} className="flex items-start space-x-4">
                <div className="flex-shrink-0 w-8 h-8 bg-primary-600 text-white rounded-full flex items-center justify-center text-sm font-bold">
                  {index + 1}
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-1">
                    {step.step}
                  </h3>
                  <p className="text-gray-600">
                    {step.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Capabilities */}
      <section>
        <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center">
          What We Support
        </h2>
        <div className="card">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {capabilities.map((capability, index) => (
              <div key={index} className="flex items-center space-x-3">
                <CheckCircle className="h-5 w-5 text-green-500 flex-shrink-0" />
                <span className="text-gray-700">{capability}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Technical Details */}
      <section>
        <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center">
          Technical Implementation
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Backend Stack
            </h3>
            <ul className="space-y-2 text-sm text-gray-600">
              <li>• FastAPI with async processing</li>
              <li>• Hugging Face Transformers</li>
              <li>• PyTorch for model inference</li>
              <li>• Redis for intelligent caching</li>
              <li>• SQLAlchemy for data persistence</li>
              <li>• Docker containerization</li>
            </ul>
          </div>
          
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              AI Models
            </h3>
            <ul className="space-y-2 text-sm text-gray-600">
              <li>• BART-large-CNN for news summarization</li>
              <li>• Pegasus for long document processing</li>
              <li>• T5-large for general text tasks</li>
              <li>• Whisper for audio transcription</li>
              <li>• Ensemble voting for quality</li>
              <li>• Confidence scoring algorithms</li>
            </ul>
          </div>
          
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Frontend Stack
            </h3>
            <ul className="space-y-2 text-sm text-gray-600">
              <li>• React 18 with hooks</li>
              <li>• Tailwind CSS for styling</li>
              <li>• Axios for API communication</li>
              <li>• React Markdown for rendering</li>
              <li>• File upload with drag & drop</li>
              <li>• Real-time streaming support</li>
            </ul>
          </div>
          
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Production Features
            </h3>
            <ul className="space-y-2 text-sm text-gray-600">
              <li>• Rate limiting and security</li>
              <li>• Health monitoring endpoints</li>
              <li>• Horizontal scaling support</li>
              <li>• Comprehensive error handling</li>
              <li>• Logging and observability</li>
              <li>• Cloud deployment ready</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Interview Ready */}
      <section className="card bg-gradient-to-r from-primary-50 to-blue-50">
        <h2 className="text-2xl font-bold text-gray-900 mb-4 text-center">
          Interview-Ready Explanation
        </h2>
        <div className="space-y-4 text-gray-700">
          <p>
            <strong>Why Hierarchical Summarization?</strong> Long documents exceed model token limits. 
            Our map-reduce approach chunks content intelligently, summarizes in parallel, then 
            hierarchically merges results for comprehensive coverage without information loss.
          </p>
          <p>
            <strong>Hallucination Control:</strong> We implement confidence scoring, source alignment 
            tracking, and ensemble voting. Each sentence is mapped to source content with confidence 
            metrics, flagging low-confidence claims for user awareness.
          </p>
          <p>
            <strong>Offline Architecture:</strong> All models run locally using Hugging Face Transformers. 
            No external API dependencies ensure data privacy, cost predictability, and reliable operation 
            in air-gapped environments.
          </p>
          <p>
            <strong>Production Scalability:</strong> Async FastAPI with Redis caching, database persistence, 
            and Docker containerization. Horizontal scaling through load balancing with health checks 
            and monitoring endpoints.
          </p>
        </div>
      </section>

      {/* Call to Action */}
      <section className="text-center">
        <div className="card bg-gray-900 text-white">
          <h2 className="text-2xl font-bold mb-4">
            Ready to Get Started?
          </h2>
          <p className="text-gray-300 mb-6">
            Deploy your own instance or contribute to the open-source project
          </p>
          <div className="flex justify-center space-x-4">
            <a
              href="/"
              className="bg-white text-gray-900 px-6 py-3 rounded-lg font-medium hover:bg-gray-100 transition-colors"
            >
              Try Demo
            </a>
            <a
              href="https://github.com/your-repo/summarization-platform"
              target="_blank"
              rel="noopener noreferrer"
              className="border border-gray-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-gray-800 transition-colors flex items-center space-x-2"
            >
              <Github className="h-5 w-5" />
              <span>Star on GitHub</span>
            </a>
          </div>
        </div>
      </section>
    </div>
  );
}

export default AboutPage;