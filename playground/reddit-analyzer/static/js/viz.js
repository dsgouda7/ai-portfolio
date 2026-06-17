const COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
    '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2'
];

let currentView = 'clusters';
let selectedClusterId = null;
let clusterData = null;
let svg = null;
let width, height;
let availableFilters = {};

const container = d3.select('#visualization');
width = container.node().clientWidth;
height = container.node().clientHeight || 600;

// API calls
async function fetchClusters() {
    const response = await fetch('/api/clusters');
    return response.json();
}

async function fetchKeywords(clusterId) {
    const response = await fetch(`/api/cluster/${clusterId}/keywords`);
    return response.json();
}

async function fetchMetadata(clusterId) {
    const response = await fetch(`/api/cluster/${clusterId}/metadata`);
    return response.json();
}

async function fetchAvailableFilters() {
    const response = await fetch('/api/available-filters');
    return response.json();
}

async function fetchKeywordCorrelations() {
    const lang = document.getElementById('languageFilter').value;
    const source = document.getElementById('sourceFilter').value;
    const timePeriod = document.getElementById('timeperiodFilter').value;

    const params = new URLSearchParams();
    if (lang !== 'all') params.append('language', lang);
    if (source !== 'all') params.append('source', source);
    if (timePeriod !== 'all') params.append('time_period', timePeriod);

    const url = `/api/keyword-correlations${params.toString() ? '?' + params.toString() : ''}`;
    const response = await fetch(url);
    return response.json();
}

// Initialize filters
async function initializeFilters() {
    try {
        availableFilters = await fetchAvailableFilters();

        // Populate language filter
        const langFilter = document.getElementById('languageFilter');
        if (availableFilters.languages) {
            availableFilters.languages.forEach(lang => {
                const option = document.createElement('option');
                option.value = lang;
                option.textContent = lang;
                langFilter.appendChild(option);
            });
        }

        // Populate source filter
        const sourceFilter = document.getElementById('sourceFilter');
        if (availableFilters.sources) {
            availableFilters.sources.forEach(source => {
                const option = document.createElement('option');
                option.value = source;
                option.textContent = source;
                sourceFilter.appendChild(option);
            });
        }

        // Populate time period filter
        const timeFilter = document.getElementById('timeperiodFilter');
        if (availableFilters.time_periods) {
            availableFilters.time_periods.forEach(period => {
                const option = document.createElement('option');
                option.value = period;
                option.textContent = period;
                timeFilter.appendChild(option);
            });
        }

        // Add event listeners to filters
        document.getElementById('languageFilter').addEventListener('change', handleFilterChange);
        document.getElementById('sourceFilter').addEventListener('change', handleFilterChange);
        document.getElementById('timeperiodFilter').addEventListener('change', handleFilterChange);

    } catch (error) {
        console.error('Error initializing filters:', error);
    }
}

async function handleFilterChange() {
    // Reload cluster data with new filters
    if (currentView === 'clusters') {
        await showClustersView();
    } else if (currentView === 'keywords' && selectedClusterId !== null) {
        await showKeywordsView(selectedClusterId);
    }
}

function createSVG() {
    container.selectAll('*').remove();
    svg = container.append('svg')
        .attr('width', width)
        .attr('height', height);
    return svg;
}

function getClusterColor(clusterId) {
    return COLORS[clusterId % COLORS.length];
}

async function showClustersView() {
    currentView = 'clusters';
    selectedClusterId = null;
    document.getElementById('info').textContent = 'Click a bubble to explore keywords and metadata';
    document.getElementById('metadataPanel').style.display = 'none';

    const data = await fetchClusters();
    const clusters = Array.isArray(data) ? data : data.clusters || [];

    createSVG();

    // Prepare data with hierarchy for bubble layout
    const hierarchyData = {
        name: 'root',
        children: clusters.map((c) => ({
            id: c.id,
            name: c.label,
            value: Math.max(20, Math.sqrt(c.count) * 10),
            count: c.count,
            color: getClusterColor(c.id)
        }))
    };

    // Create hierarchy
    const root = d3.hierarchy(hierarchyData)
        .sum(d => d.value);

    // Create pack layout
    const pack = d3.pack()
        .size([width, height])
        .padding(3);

    const nodes = pack(root).leaves();

    // Create bubbles
    svg.selectAll('.bubble')
        .data(nodes, d => d.data.id)
        .enter()
        .append('circle')
        .attr('class', 'bubble')
        .attr('cx', d => d.x)
        .attr('cy', d => d.y)
        .attr('r', d => d.r)
        .attr('fill', d => d.data.color)
        .attr('opacity', 0.8)
        .on('mouseover', function() {
            d3.select(this).attr('opacity', 1);
        })
        .on('mouseout', function() {
            d3.select(this).attr('opacity', 0.8);
        })
        .on('click', (event, d) => {
            showKeywordsView(d.data.id);
        });

    // Add labels
    svg.selectAll('.bubble-label')
        .data(nodes, d => `label-${d.data.id}`)
        .enter()
        .append('text')
        .attr('class', 'bubble-label')
        .attr('x', d => d.x)
        .attr('y', d => d.y - 10)
        .attr('font-size', d => Math.max(10, d.r / 4))
        .text(d => {
            const text = d.data.name;
            const maxLength = Math.floor(d.r / 4);
            return text.length > maxLength ? text.substring(0, maxLength - 1) + '…' : text;
        });

    // Add count
    svg.selectAll('.bubble-count')
        .data(nodes, d => `count-${d.data.id}`)
        .enter()
        .append('text')
        .attr('class', 'bubble-count')
        .attr('x', d => d.x)
        .attr('y', d => d.y + 10)
        .text(d => `${d.data.count} comments`);
}

async function showKeywordsView(clusterId) {
    currentView = 'keywords';
    selectedClusterId = clusterId;

    const keywordsData = await fetchKeywords(clusterId);
    const keywords = keywordsData.keywords || [];

    // Load and display metadata
    try {
        const metadata = await fetchMetadata(clusterId);
        displayMetadata(metadata);
    } catch (error) {
        console.error('Error loading metadata:', error);
    }

    document.getElementById('info').textContent = `Showing keywords for Cluster ${clusterId}. Click reset to go back.`;

    createSVG();

    // Prepare data
    const maxCount = Math.max(...keywords.map(k => k.count), 1);
    const hierarchyData = {
        name: 'root',
        children: keywords.map(k => ({
            word: k.word,
            value: Math.max(5, (k.count / maxCount) * 50),
            count: k.count
        }))
    };

    // Create hierarchy
    const root = d3.hierarchy(hierarchyData)
        .sum(d => d.value);

    // Create pack layout
    const pack = d3.pack()
        .size([width, height])
        .padding(2);

    const nodes = pack(root).leaves();

    // Create bubbles
    svg.selectAll('.keyword-bubble')
        .data(nodes)
        .enter()
        .append('circle')
        .attr('class', 'keyword-bubble')
        .attr('cx', d => d.x)
        .attr('cy', d => d.y)
        .attr('r', d => d.r)
        .attr('fill', getClusterColor(clusterId))
        .attr('opacity', 0.7)
        .on('mouseover', function(event, d) {
            d3.select(this).attr('opacity', 1);
            showTooltip(event, d.data.word, d.data.count);
        })
        .on('mouseout', function() {
            d3.select(this).attr('opacity', 0.7);
            hideTooltip();
        });

    // Add labels
    svg.selectAll('.keyword-label')
        .data(nodes)
        .enter()
        .append('text')
        .attr('class', 'keyword-label')
        .attr('x', d => d.x)
        .attr('y', d => d.y)
        .attr('font-size', d => Math.max(8, d.r / 3))
        .text(d => {
            const text = d.data.word;
            const maxLength = Math.max(1, Math.floor(d.r / 3));
            return text.length > maxLength ? text.substring(0, maxLength - 1) + '…' : text;
        });
}

function displayMetadata(metadata) {
    const panel = document.getElementById('metadataPanel');
    const content = document.getElementById('metadataContent');

    content.innerHTML = '';

    // Format numbers
    const fmt = (n) => typeof n === 'number' ? Math.round(n * 100) / 100 : n;

    // Cluster summary
    const summaryHTML = `
        <div class="metadata-section">
            <h3>Cluster Summary</h3>
            <ul class="metadata-list">
                <li>
                    <span class="metadata-label">Total Comments</span>
                    <span class="metadata-value">${metadata.count}</span>
                </li>
                <li>
                    <span class="metadata-label">Avg Sentiment</span>
                    <span class="metadata-value">${fmt(metadata.avg_sentiment)}</span>
                </li>
                <li>
                    <span class="metadata-label">Avg Score</span>
                    <span class="metadata-value">${metadata.avg_score}</span>
                </li>
                <li>
                    <span class="metadata-label">Avg Depth</span>
                    <span class="metadata-value">${metadata.avg_depth}</span>
                </li>
            </ul>
        </div>
    `;
    content.insertAdjacentHTML('beforeend', summaryHTML);

    // Sources distribution
    if (metadata.sources && Object.keys(metadata.sources).length > 0) {
        let sourcesHTML = '<div class="metadata-section"><h3>Sources</h3><ul class="metadata-list">';
        Object.entries(metadata.sources).slice(0, 5).forEach(([source, count]) => {
            sourcesHTML += `<li><span class="metadata-label">${source}</span><span class="metadata-value">${count}</span></li>`;
        });
        sourcesHTML += '</ul></div>';
        content.insertAdjacentHTML('beforeend', sourcesHTML);
    }

    // Languages distribution
    if (metadata.languages && Object.keys(metadata.languages).length > 0) {
        let langsHTML = '<div class="metadata-section"><h3>Languages</h3><ul class="metadata-list">';
        Object.entries(metadata.languages).forEach(([lang, count]) => {
            langsHTML += `<li><span class="metadata-label">${lang}</span><span class="metadata-value">${count}</span></li>`;
        });
        langsHTML += '</ul></div>';
        content.insertAdjacentHTML('beforeend', langsHTML);
    }

    // Time periods distribution
    if (metadata.time_periods && Object.keys(metadata.time_periods).length > 0) {
        let timeHTML = '<div class="metadata-section"><h3>Time Periods</h3><ul class="metadata-list">';
        Object.entries(metadata.time_periods).forEach(([period, count]) => {
            timeHTML += `<li><span class="metadata-label">${period}</span><span class="metadata-value">${count}</span></li>`;
        });
        timeHTML += '</ul></div>';
        content.insertAdjacentHTML('beforeend', timeHTML);
    }

    // Engagement tiers
    if (metadata.engagement_tiers && Object.keys(metadata.engagement_tiers).length > 0) {
        let engHTML = '<div class="metadata-section"><h3>Engagement</h3><ul class="metadata-list">';
        Object.entries(metadata.engagement_tiers).forEach(([tier, count]) => {
            engHTML += `<li><span class="metadata-label">${tier}</span><span class="metadata-value">${count}</span></li>`;
        });
        engHTML += '</ul></div>';
        content.insertAdjacentHTML('beforeend', engHTML);
    }

    panel.style.display = 'block';
}

function showTooltip(event, word, count) {
    const tooltip = document.querySelector('.tooltip') || document.createElement('div');
    tooltip.className = 'tooltip';
    tooltip.textContent = `${word} (${count})`;
    document.body.appendChild(tooltip);

    const x = event.pageX + 10;
    const y = event.pageY + 10;
    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
}

function hideTooltip() {
    const tooltip = document.querySelector('.tooltip');
    if (tooltip) {
        tooltip.remove();
    }
}

// Event listeners
document.getElementById('resetBtn').addEventListener('click', showClustersView);

// Initialize filters and load clusters
(async () => {
    await initializeFilters();
    await showClustersView();
})();

// Handle window resize
window.addEventListener('resize', () => {
    width = container.node().clientWidth;
    height = container.node().clientHeight || 600;

    if (currentView === 'clusters') {
        showClustersView();
    } else if (currentView === 'keywords' && selectedClusterId !== null) {
        showKeywordsView(selectedClusterId);
    }
});
