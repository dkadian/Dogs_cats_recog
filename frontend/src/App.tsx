import { useState, useEffect } from 'react';
import type { ChangeEvent, DragEvent } from 'react';
import './App.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5050';

interface Prediction {
  category?: string;
  confidence: number;
  details?: string;
  is_dog?: boolean;
  is_cat?: boolean;
  alternatives?: string[];
  info?: {
    summary?: string;
    image?: string;
    wiki_url?: string;
    origin?: string;
    temperament?: string;
    life_span?: string;
    weight?: string;
    source?: string;
  };
}

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedImageUrl, setSelectedImageUrl] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState<boolean>(false);

  // Semi-circle Odometer calculations
  const radius = 50;
  const arcLength = Math.PI * radius; 
  const arcOffset = prediction ? arcLength - (prediction.confidence * arcLength) : arcLength;

  // Global drag prevention to ensure drops are caught correctly
  useEffect(() => {
    const preventDefaults = (e: Event) => {
      e.preventDefault();
      e.stopPropagation();
    };

    window.addEventListener('dragover', preventDefaults);
    window.addEventListener('drop', preventDefaults);

    return () => {
      window.removeEventListener('dragover', preventDefaults);
      window.removeEventListener('drop', preventDefaults);
    };
  }, []);

  const processFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError("Please provide a valid image file (PNG, JPG, etc.)");
      return;
    }
    
    setError(null);
    setPrediction(null);
    setSelectedFile(file);
    setSelectedImageUrl(null);

    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target?.result as string);
    };
    reader.onerror = () => {
      setError("Failed to read the image file. Please try again.");
    };
    reader.readAsDataURL(file);
  };

  const getImageUrlFromDrop = (dataTransfer: DataTransfer) => {
    const html = dataTransfer.getData('text/html');
    if (html) {
      const documentFragment = new DOMParser().parseFromString(html, 'text/html');
      const image = documentFragment.querySelector('img');
      const src = image?.getAttribute('src') || image?.getAttribute('data-src');

      if (src) {
        return src;
      }
    }

    const uriList = dataTransfer.getData('text/uri-list');
    if (uriList) {
      const uri = uriList
        .split('\n')
        .map((line) => line.trim())
        .find((line) => line && !line.startsWith('#'));

      if (uri) {
        return uri;
      }
    }

    const text = dataTransfer.getData('text/plain').trim();
    if (text) {
      return text;
    }

    return null;
  };

  const makeImageNameFromUrl = (imageUrl: string) => {
    try {
      const url = new URL(imageUrl);
      const name = url.pathname.split('/').filter(Boolean).pop();
      return name || url.hostname;
    } catch {
      return 'Dropped image';
    }
  };

  const processImageUrl = async (imageUrl: string) => {
    if (imageUrl.startsWith('data:image/')) {
      try {
        const response = await fetch(imageUrl);
        const blob = await response.blob();
        processFile(new File([blob], 'dropped-image.png', { type: blob.type || 'image/png' }));
      } catch {
        setError('Failed to read the dropped image. Please try saving it and uploading the file.');
      }
      return;
    }

    try {
      const parsedUrl = new URL(imageUrl);
      if (!['http:', 'https:'].includes(parsedUrl.protocol)) {
        throw new Error('Unsupported image URL');
      }

      setError(null);
      setPrediction(null);
      setSelectedFile(null);
      setSelectedImageUrl(parsedUrl.toString());
      setPreview(parsedUrl.toString());
    } catch {
      setError('The dropped item was not an image. Please drop an image file or image from a web page.');
    }
  };

  const handleDrag = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragover' || e.type === 'dragenter') {
      setDragging(true);
    } else if (e.type === 'dragleave') {
      setDragging(false);
    }
  };

  const handleDrop = async (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
      return;
    }

    const imageUrl = getImageUrlFromDrop(e.dataTransfer);
    if (imageUrl) {
      await processImageUrl(imageUrl);
      return;
    }

    setError('No image was found in the dropped item. Try opening the image first, then drag it here.');
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0]);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile && !selectedImageUrl) return;
    setLoading(true);
    setError(null);
    const formData = new FormData();
    if (selectedFile) {
      formData.append('image', selectedFile);
    } else if (selectedImageUrl) {
      formData.append('image_url', selectedImageUrl);
    }
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Prediction engine encountered an error.");
      }
      const data: Prediction = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred during analysis.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-root">
      <div className="bg-blobs">
        <div className="blob blob-1"></div>
        <div className="blob blob-2"></div>
        <div className="blob blob-3"></div>
      </div>

      <div className="container">
        <header>
          <h1>Pet Identifier</h1>
          <p>Discover your pet's breed with high-precision AI.</p>
        </header>

        <div className="glass-card upload-section">
          {/* Hidden Input Moved Outside for cleaner event flow */}
          <input 
            type="file" 
            accept="image/*" 
            onChange={handleFileChange} 
            id="file-input"
            className="file-input"
            style={{ display: 'none' }}
          />
          
          <div 
            className={`drop-zone ${dragging ? 'dragging' : ''}`}
            onDragOver={handleDrag}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDrop={handleDrop}
            onClick={() => document.getElementById('file-input')?.click()}
          >
            <div className="icon-upload">↑</div>
            <div className="file-label">
              {selectedFile ? selectedFile.name : (selectedImageUrl ? makeImageNameFromUrl(selectedImageUrl) : "Drop a pet photo or click to browse")}
            </div>
          </div>
          
          <button 
            onClick={handlePredict} 
            disabled={(!selectedFile && !selectedImageUrl) || loading}
            className="predict-button"
          >
            {loading ? "Analyzing Specimen..." : "Identify Breed"}
          </button>
        </div>

        {error && <div className="error-toast">{error}</div>}

        <div className="result-grid">
          <div className={`glass-card result-card ${prediction ? 'animate-in delay-1' : ''}`}>
            <h2>Image Preview</h2>
            <div className="preview-container">
              {preview ? (
                <img src={preview} alt="Preview" className="preview-img" />
              ) : (
                <div className="placeholder-box" style={{height: '100%'}}>Ready for scanning</div>
              )}
            </div>
          </div>

          <div className={`glass-card result-card ${prediction ? 'animate-in delay-2' : ''}`}>
            <h2>Identification</h2>
            {prediction ? (
              <div className="identification-content">
                <div className="breed-header">
                  <div className="breed-name">{prediction.details}</div>
                  <div className="breed-category">
                    {prediction.is_dog ? "Canine Breed" : (prediction.is_cat ? "Feline Breed" : "Unknown Category")}
                  </div>
                  {prediction.alternatives && prediction.alternatives.length > 0 && (
                    <div className="breed-alternatives">
                      Also compare: {prediction.alternatives.join(', ')}
                    </div>
                  )}
                </div>

                <div className="gauge-area">
                  <svg className="gauge-svg" viewBox="0 0 120 70">
                    <defs>
                      <linearGradient id="gauge-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#6366f1" />
                        <stop offset="100%" stopColor="#a855f7" />
                      </linearGradient>
                    </defs>
                    <path className="gauge-track" d="M 10,60 A 50,50 0 0 1 110,60" />
                    <path 
                      className="gauge-value-fill" 
                      d="M 10,60 A 50,50 0 0 1 110,60" 
                      style={{ 
                        strokeDasharray: arcLength, 
                        strokeDashoffset: arcOffset 
                      }} 
                    />
                  </svg>
                  <div className="gauge-percentage-inside">{(prediction.confidence * 100).toFixed(0)}%</div>
                </div>
                <div className="gauge-label-below">Match confidence</div>
              </div>
            ) : (
              <div className="placeholder-box" style={{height: '100%'}}>
                {loading ? "AI is processing data..." : "Awaiting specimen"}
              </div>
            )}
          </div>

          {prediction && (prediction.is_dog || prediction.is_cat) && (
            <div className="info-row glass-card animate-in delay-3">
              <div className="wiki-header">
                <h2>Breed Encyclopedia</h2>
                <span className="source-badge">{prediction.info?.source || "Encyclopedia"}</span>
              </div>

              <div className="wiki-content-layout">
                {prediction.info?.image && (
                  <div className="wiki-thumb-container">
                    <img src={prediction.info.image} alt={prediction.details} className="wiki-thumb" />
                  </div>
                )}
                <div className="summary-text">
                  {prediction.info?.summary || "No detailed summary available for this specific specimen."}
                </div>
              </div>

              <div className="spec-grid">
                <div className="spec-box">
                  <span className="spec-label">Temperament</span>
                  <span className="spec-value">{prediction.info?.temperament || "Varies"}</span>
                </div>
                <div className="spec-box">
                  <span className="spec-label">Life Span</span>
                  <span className="spec-value">{prediction.info?.life_span || "10-15 years"}</span>
                </div>
                <div className="spec-box">
                  <span className="spec-label">Weight</span>
                  <span className="spec-value">{prediction.info?.weight || "Varies"}</span>
                </div>
                <div className="spec-box">
                  <span className="spec-label">Region of Origin</span>
                  <span className="spec-value">{prediction.info?.origin || "International"}</span>
                </div>
              </div>
            </div>
          )}
        </div>

        <footer>
          &copy; 2026 Pet Identifier AI • Premium Edition
        </footer>
      </div>
    </div>
  );
}

export default App;
