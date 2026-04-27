let analysisData = {};
let selectedCoin = null;
let selectedInterval = "1h";
const selectedRange = "6m";

let priceChart = null;
let liveSocket = null;
let liveReconnectTimer = null;
let liveStreamCoin = null;

const INTERVAL_OPTIONS = ["1h", "4h"];
const DISPLAY_TIMEZONE = "Asia/Kuala_Lumpur";
const MYT_OFFSET_MS = 8 * 60 * 60 * 1000;

const rawHistoryCache = {};
let horizontalPanBound = false;
let dragPanBound = false;
let isDraggingChart = false;
let dragStartX = 0;
let viewportState = {
    min: null,
    max: null,
    defaultMin: null,
    defaultMax: null
};
let interactionLockUntil = 0;

function nowMs() {
    return Date.now();
}

function isInteractionLocked() {
    return nowMs() < interactionLockUntil;
}

function lockChartInteraction(ms = 1800) {
    interactionLockUntil = nowMs() + ms;
}

function getRawHistoryCacheKey(coin) {
    return `${coin}|1h|${selectedRange}`;
}

function formatPrice(value) {
    if (value === null || value === undefined || isNaN(value)) return "-";

    if (value >= 1000) {
        return `$${Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
    }

    if (value >= 1) {
        return `$${Number(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 })}`;
    }

    return `$${Number(value).toLocaleString(undefined, { minimumFractionDigits: 4, maximumFractionDigits: 6 })}`;
}

function formatPercent(value, digits = 2) {
    if (value === null || value === undefined || isNaN(value)) return "-";
    const sign = Number(value) > 0 ? "+" : "";
    return `${sign}${Number(value).toFixed(digits)}%`;
}

function formatPlain(value, digits = 2) {
    if (value === null || value === undefined || isNaN(value)) return "-";
    return Number(value).toFixed(digits);
}

function signalClass(signal) {
    if (signal === "BUY") return "signal-buy";
    if (signal === "SELL") return "signal-sell";
    return "signal-hold";
}

function sentimentClass(label) {
    if (label === "BULLISH") return "sentiment-bullish";
    if (label === "BEARISH") return "sentiment-bearish";
    return "sentiment-neutral";
}

function mtfTone(value) {
    if (value === "UP") return "text-green-700";
    if (value === "DOWN") return "text-rose-700";
    return "text-gray-700";
}

function listToHtml(items, type) {
    if (!items || !items.length) {
        return `<li class="flex gap-3"><span class="mt-1 w-2 h-2 rounded-full bg-gray-300"></span><span>No data available.</span></li>`;
    }

    const marker =
        type === "bull"
            ? "bg-green-500"
            : type === "bear"
              ? "bg-rose-500"
              : "bg-amber-500";

    return items
        .map(item => `
            <li class="flex gap-3">
                <span class="mt-2 w-2 h-2 rounded-full ${marker} shrink-0"></span>
                <span>${item}</span>
            </li>
        `)
        .join("");
}

function getBinancePair(coin) {
    return `${coin.toLowerCase()}usdt`;
}

function setConnectionStatus(text, type = "neutral") {
    const statusEl = document.getElementById("connectionStatus");

    if (type === "success") {
        statusEl.textContent = text;
        statusEl.className = "px-3 py-2 rounded-full border border-green-200 bg-green-50 text-sm font-medium text-green-800";
        return;
    }

    if (type === "warning") {
        statusEl.textContent = text;
        statusEl.className = "px-3 py-2 rounded-full border border-amber-200 bg-amber-50 text-sm font-medium text-amber-800";
        return;
    }

    if (type === "error") {
        statusEl.textContent = text;
        statusEl.className = "px-3 py-2 rounded-full border border-red-200 bg-red-50 text-sm font-medium text-red-800";
        return;
    }

    statusEl.textContent = text;
    statusEl.className = "px-3 py-2 rounded-full border border-line bg-white text-sm font-medium text-gray-700";
}

function updateLastUpdatedLabel() {
    const label = new Date().toLocaleString(undefined, {
        timeZone: DISPLAY_TIMEZONE,
        day: "numeric",
        month: "short",
        hour: "numeric",
        minute: "2-digit",
        hour12: true
    });
    document.getElementById("lastUpdated").textContent = `Updated ${label} MYT`;
}

function buildCoinSelector(coins) {
    const container = document.getElementById("coinSelector");
    container.innerHTML = "";

    coins.forEach(coin => {
        const button = document.createElement("button");
        button.className = "coin-chip px-4 py-2.5 rounded-full border border-line bg-white text-sm font-semibold transition-all whitespace-nowrap hover:border-ink";
        button.textContent = coin;

        if (coin === selectedCoin) {
            button.classList.add("active");
        }

        button.addEventListener("click", async () => {
            selectedCoin = coin;
            buildCoinSelector(Object.keys(analysisData));
            renderCoin(coin);
            connectLiveStream(coin);
            await loadHistoryAndRenderChart(true);
        });

        container.appendChild(button);
    });
}

function buildIntervalSelector() {
    const container = document.getElementById("intervalSelector");
    container.innerHTML = "";

    INTERVAL_OPTIONS.forEach(interval => {
        const button = document.createElement("button");
        const active = interval === selectedInterval;

        button.className = active
            ? "filter-chip active px-3 py-1.5 rounded-full border text-sm font-semibold"
            : "filter-chip px-3 py-1.5 rounded-full border border-line bg-white text-sm font-semibold";

        button.textContent = interval.toUpperCase();

        button.addEventListener("click", async () => {
            selectedInterval = interval;
            buildIntervalSelector();
            updateViewMeta();
            connectLiveStream(selectedCoin);
            await loadHistoryAndRenderChart(true);
        });

        container.appendChild(button);
    });
}

function updateViewMeta() {
    document.getElementById("viewMeta").textContent = `${selectedCoin || "-"} · ${selectedInterval} · 6M`;
    document.getElementById("chartMeta").textContent = `Binance 1h history aggregated to ${selectedInterval} · 6M`;
}

function formatHistoryTooltip(ts) {
    return new Date(ts).toLocaleString(undefined, {
        timeZone: DISPLAY_TIMEZONE,
        weekday: "short",
        day: "numeric",
        month: "short",
        year: "numeric",
        hour: "numeric",
        minute: "2-digit",
        hour12: true
    }) + " (MYT)";
}

function getTimeUnitForInterval(interval) {
    if (interval === "4h") return "day";
    return "hour";
}

function getIntervalMs(interval) {
    if (interval === "4h") return 4 * 60 * 60 * 1000;
    return 60 * 60 * 1000;
}

function bucketStartMsForInterval(tsMs, interval) {
    if (interval === "1h") {
        return tsMs;
    }

    const shifted = tsMs + MYT_OFFSET_MS;

    if (interval === "4h") {
        const bucket = Math.floor(shifted / (4 * 60 * 60 * 1000)) * (4 * 60 * 60 * 1000);
        return bucket - MYT_OFFSET_MS;
    }

    const bucket = Math.floor(shifted / (24 * 60 * 60 * 1000)) * (24 * 60 * 60 * 1000);
    return bucket - MYT_OFFSET_MS;
}

function aggregateCandlesForDisplay(rawCandles, interval) {
    if (!rawCandles || !rawCandles.length) return [];

    if (interval === "1h") {
        return rawCandles.map(item => ({
            time: Number(item.time) * 1000,
            open: Number(item.open),
            high: Number(item.high),
            low: Number(item.low),
            close: Number(item.close),
            volume: Number(item.volume || 0)
        }));
    }

    const buckets = new Map();

    rawCandles.forEach(item => {
        const tsMs = Number(item.time) * 1000;
        const bucketStart = bucketStartMsForInterval(tsMs, interval);

        if (!buckets.has(bucketStart)) {
            buckets.set(bucketStart, {
                time: bucketStart,
                open: Number(item.open),
                high: Number(item.high),
                low: Number(item.low),
                close: Number(item.close),
                volume: Number(item.volume || 0)
            });
        } else {
            const bucket = buckets.get(bucketStart);
            bucket.high = Math.max(bucket.high, Number(item.high));
            bucket.low = Math.min(bucket.low, Number(item.low));
            bucket.close = Number(item.close);
            bucket.volume += Number(item.volume || 0);
        }
    });

    return Array.from(buckets.values()).sort((a, b) => a.time - b.time);
}

function buildForecastDisplayData(historyTimestamps, forecastValues, analysisPayload, interval) {
    if (!forecastValues || !forecastValues.length || !historyTimestamps.length) {
        return {
            basePoints: [],
            lowerPoints: [],
            upperPoints: []
        };
    }

    const currentPrice = Number(analysisPayload?.current_price || 0);
    const atr = Number(analysisPayload?.technical_context?.atr || 0);
    const volatility = Number(analysisPayload?.volatility || 0);

    const baseBandPct = Math.max(
        currentPrice > 0 ? (atr / currentPrice) * 0.45 : 0,
        volatility * 1.25,
        0.0018
    );

    const lastTs = historyTimestamps[historyTimestamps.length - 1];
    const intervalMs = getIntervalMs(interval);

    const selectedSteps =
        interval === "4h"
            ? [4, 8, 12, 16, 20, 24].filter(step => forecastValues[step - 1] !== undefined)
            : Array.from({ length: Math.min(forecastValues.length, 24) }, (_, i) => i + 1);

    const basePoints = [];
    const lowerPoints = [];
    const upperPoints = [];

    selectedSteps.forEach((step, visualIndex) => {
        const rawValue = Number(forecastValues[step - 1]);
        const wave =
            (Math.sin((visualIndex + 1) * 0.9) * 0.0022) +
            (Math.cos((visualIndex + 1) * 0.45) * 0.0014);

        const adjustedValue = rawValue * (1 + wave);
        const bandPct = baseBandPct * (1 + (visualIndex + 1) * 0.08);
        const bandValue = adjustedValue * bandPct;

        const x = lastTs + ((visualIndex + 1) * intervalMs);

        basePoints.push({ x, y: adjustedValue });
        lowerPoints.push({ x, y: adjustedValue - bandValue });
        upperPoints.push({ x, y: adjustedValue + bandValue });
    });

    return {
        basePoints,
        lowerPoints,
        upperPoints
    };
}

function saveViewportIfAvailable() {
    if (!priceChart || !priceChart.scales || !priceChart.scales.x) return;

    const xScale = priceChart.scales.x;
    if (xScale.min === undefined || xScale.max === undefined) return;

    viewportState.min = xScale.min;
    viewportState.max = xScale.max;
}

function setViewport(min, max) {
    if (!priceChart) return;

    priceChart.options.scales.x.min = min;
    priceChart.options.scales.x.max = max;
    priceChart.update("none");

    viewportState.min = min;
    viewportState.max = max;
}

function resetChartToDefaultView() {
    if (!priceChart) return;
    setViewport(viewportState.defaultMin, viewportState.defaultMax);
}

function panViewport(deltaMs) {
    if (!priceChart || !priceChart.scales || !priceChart.scales.x) return;

    const xScale = priceChart.scales.x;
    const currentMin = xScale.min ?? viewportState.defaultMin;
    const currentMax = xScale.max ?? viewportState.defaultMax;

    const range = currentMax - currentMin;
    let nextMin = currentMin + deltaMs;
    let nextMax = currentMax + deltaMs;

    if (nextMin < viewportState.defaultMin) {
        nextMin = viewportState.defaultMin;
        nextMax = nextMin + range;
    }

    if (nextMax > viewportState.defaultMax) {
        nextMax = viewportState.defaultMax;
        nextMin = nextMax - range;
    }

    setViewport(nextMin, nextMax);
    lockChartInteraction();
}

function zoomViewport(centerX, factor) {
    if (!priceChart || !priceChart.scales || !priceChart.scales.x) return;

    const xScale = priceChart.scales.x;
    const currentMin = xScale.min ?? viewportState.defaultMin;
    const currentMax = xScale.max ?? viewportState.defaultMax;

    const currentRange = Math.max(currentMax - currentMin, getIntervalMs(selectedInterval) * 8);
    let nextRange = currentRange * factor;

    const minRange = getIntervalMs(selectedInterval) * 8;
    const maxRange = viewportState.defaultMax - viewportState.defaultMin;

    nextRange = Math.max(minRange, Math.min(maxRange, nextRange));

    const ratio = (centerX - currentMin) / currentRange;
    let nextMin = centerX - (nextRange * ratio);
    let nextMax = nextMin + nextRange;

    if (nextMin < viewportState.defaultMin) {
        nextMin = viewportState.defaultMin;
        nextMax = nextMin + nextRange;
    }

    if (nextMax > viewportState.defaultMax) {
        nextMax = viewportState.defaultMax;
        nextMin = nextMax - nextRange;
    }

    setViewport(nextMin, nextMax);
    lockChartInteraction();
}

function bindHorizontalTrackpadPan() {
    const canvas = document.getElementById("priceChart");
    if (!canvas || horizontalPanBound) return;

    horizontalPanBound = true;

    canvas.addEventListener("wheel", (event) => {
        if (!priceChart) return;

        const rect = canvas.getBoundingClientRect();
        const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
        const xScale = priceChart.scales.x;
        const currentMin = xScale.min ?? viewportState.defaultMin;
        const currentMax = xScale.max ?? viewportState.defaultMax;
        const centerX = currentMin + ((currentMax - currentMin) * ratio);

        if (Math.abs(event.deltaX) > Math.abs(event.deltaY) && Math.abs(event.deltaX) > 0) {
            event.preventDefault();
            const panStep = (currentMax - currentMin) * 0.08;
            panViewport(event.deltaX > 0 ? panStep : -panStep);
            return;
        }

        event.preventDefault();
        const zoomFactor = event.deltaY > 0 ? 1.12 : 0.88;
        zoomViewport(centerX, zoomFactor);
    }, { passive: false });
}

function bindMouseDragPan() {
    const canvas = document.getElementById("priceChart");
    if (!canvas || dragPanBound) return;

    dragPanBound = true;

    canvas.addEventListener("mousedown", (event) => {
        if (!priceChart || event.button !== 0) return;

        isDraggingChart = true;
        dragStartX = event.clientX;
        lockChartInteraction();
    });

    window.addEventListener("mousemove", (event) => {
        if (!isDraggingChart || !priceChart) return;

        const dx = event.clientX - dragStartX;
        if (Math.abs(dx) < 4) return;

        const xScale = priceChart.scales.x;
        const visibleRange = (xScale.max ?? viewportState.defaultMax) - (xScale.min ?? viewportState.defaultMin);
        const msPerPixel = visibleRange / priceChart.width;
        const deltaMs = -dx * msPerPixel;

        panViewport(deltaMs);
        dragStartX = event.clientX;
    });

    window.addEventListener("mouseup", () => {
        isDraggingChart = false;
    });

    window.addEventListener("mouseleave", () => {
        isDraggingChart = false;
    });
}

function renderChart(rawHistoryPayload, analysisPayload, preserveViewport = true) {
    const ctx = document.getElementById("priceChart").getContext("2d");
    const rawCandles = rawHistoryPayload?.candles || [];
    const displayCandles = aggregateCandlesForDisplay(rawCandles, selectedInterval);
    const forecast = analysisPayload?.forecast_next_hours || [];

    const historyPoints = displayCandles.map(item => ({
        x: item.time,
        y: Number(item.close)
    }));

    const historyTimestamps = historyPoints.map(item => item.x);

    const forecastData = buildForecastDisplayData(
        historyTimestamps,
        forecast,
        analysisPayload,
        selectedInterval
    );

    const historySeries = historyPoints;
    const historyLastPoint = historyPoints.length ? [historyPoints[historyPoints.length - 1]] : [];

    const forecastLineSeries = historyLastPoint.concat(forecastData.basePoints);
    const forecastLowerSeries = historyLastPoint.concat(forecastData.lowerPoints);
    const forecastUpperSeries = historyLastPoint.concat(forecastData.upperPoints);

    if (priceChart && preserveViewport) {
        saveViewportIfAvailable();
    }

    if (priceChart) {
        priceChart.destroy();
    }

    const fullMin = historyPoints.length ? historyPoints[0].x : Date.now();
    const fullMax = forecastData.basePoints.length
        ? forecastData.basePoints[forecastData.basePoints.length - 1].x
        : (historyPoints.length ? historyPoints[historyPoints.length - 1].x : Date.now());

    viewportState.defaultMin = fullMin;
    viewportState.defaultMax = fullMax;

    let initialMin = fullMin;
    let initialMax = fullMax;

    if (preserveViewport && viewportState.min !== null && viewportState.max !== null) {
        initialMin = Math.max(fullMin, viewportState.min);
        initialMax = Math.min(fullMax, viewportState.max);

        if (initialMax <= initialMin) {
            initialMin = fullMin;
            initialMax = fullMax;
        }
    }

    priceChart = new Chart(ctx, {
        type: "line",
        data: {
            datasets: [
                {
                    label: `${selectedCoin} Price`,
                    data: historySeries,
                    parsing: false,
                    borderColor: "#111827",
                    backgroundColor: "rgba(17, 24, 39, 0.05)",
                    borderWidth: 2.4,
                    pointRadius: 0,
                    tension: 0.22,
                    fill: false
                },
                {
                    label: "Forecast Lower",
                    data: forecastLowerSeries,
                    parsing: false,
                    borderColor: "rgba(22, 163, 74, 0)",
                    backgroundColor: "rgba(22, 163, 74, 0.10)",
                    borderWidth: 0,
                    pointRadius: 0,
                    tension: 0.24,
                    fill: false
                },
                {
                    label: "Forecast Range",
                    data: forecastUpperSeries,
                    parsing: false,
                    borderColor: "rgba(22, 163, 74, 0)",
                    backgroundColor: "rgba(22, 163, 74, 0.12)",
                    borderWidth: 0,
                    pointRadius: 0,
                    tension: 0.24,
                    fill: "-1"
                },
                {
                    label: "AI Forecast",
                    data: forecastLineSeries,
                    parsing: false,
                    borderColor: "#16a34a",
                    backgroundColor: "rgba(22, 163, 74, 0)",
                    borderWidth: 2.4,
                    pointRadius: 0,
                    borderDash: [7, 5],
                    tension: 0.24,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            interaction: {
                mode: "nearest",
                intersect: false
            },
            plugins: {
                legend: {
                    position: "top",
                    labels: {
                        color: "#374151",
                        usePointStyle: true,
                        boxWidth: 10,
                        font: {
                            size: 12,
                            weight: "600"
                        },
                        filter: function(item) {
                            return item.text !== "Forecast Lower" && item.text !== "Forecast Range";
                        }
                    }
                },
                tooltip: {
                    backgroundColor: "#111827",
                    titleColor: "#ffffff",
                    bodyColor: "#ffffff",
                    displayColors: false,
                    callbacks: {
                        title: function(items) {
                            if (!items || !items.length) return "";
                            const xValue = items[0].raw?.x;
                            return xValue ? formatHistoryTooltip(xValue) : "";
                        },
                        label: function(context) {
                            const label = context.dataset.label || "";
                            const value = context.raw?.y;

                            if (label === "Forecast Lower" || label === "Forecast Range") {
                                return null;
                            }

                            if (label === "AI Forecast") {
                                const x = context.raw?.x;
                                const lower = forecastLowerSeries.find(p => p.x === x)?.y;
                                const upper = forecastUpperSeries.find(p => p.x === x)?.y;

                                if (lower !== undefined && upper !== undefined) {
                                    return [
                                        `Expected: ${formatPrice(value)}`,
                                        `Range: ${formatPrice(lower)} - ${formatPrice(upper)}`
                                    ];
                                }
                            }

                            return `${label}: ${formatPrice(value)}`;
                        }
                    }
                },
                zoom: {
                    pan: { enabled: false },
                    zoom: {
                        wheel: { enabled: false },
                        pinch: { enabled: false },
                        drag: { enabled: false },
                        mode: "x"
                    }
                }
            },
            scales: {
                x: {
                    type: "time",
                    min: initialMin,
                    max: initialMax,
                    time: {
                        unit: getTimeUnitForInterval(selectedInterval),
                        tooltipFormat: "dd MMM yyyy HH:mm",
                        displayFormats: {
                            hour: "d MMM h a",
                            day: "d MMM"
                        }
                    },
                    grid: {
                        color: "rgba(229, 231, 235, 0.7)"
                    },
                    ticks: {
                        color: "#6b7280",
                        autoSkip: true,
                        maxTicksLimit: 10
                    }
                },
                y: {
                    grid: {
                        color: "rgba(229, 231, 235, 0.7)"
                    },
                    ticks: {
                        color: "#6b7280",
                        callback: function(value) {
                            return formatPrice(value);
                        }
                    }
                }
            }
        }
    });
}

function renderNewsArticles(articles) {
    const container = document.getElementById("newsArticles");
    container.innerHTML = "";

    if (!articles || !articles.length) {
        container.innerHTML = `
            <div class="rounded-3xl border border-line bg-shell p-5">
                <p class="text-sm text-gray-600">No relevant articles were available for this asset in the current cycle.</p>
            </div>
        `;
        return;
    }

    articles.slice(0, 3).forEach(article => {
        const badgeClass = sentimentClass(article.sentiment_label || "NEUTRAL");

        const card = document.createElement("article");
        card.className = "rounded-3xl border border-line bg-white p-5";

        card.innerHTML = `
            <div class="flex flex-wrap items-center gap-3 mb-3">
                <span class="px-2.5 py-1 rounded-full text-xs font-bold ${badgeClass}">
                    ${article.sentiment_label || "NEUTRAL"}
                </span>
                <span class="text-xs font-semibold uppercase tracking-[0.16em] text-gray-500">
                    ${article.source || "Unknown Source"}
                </span>
                <span class="text-xs text-gray-500">
                    Score: ${formatPlain(article.sentiment_score || 0, 3)}
                </span>
            </div>

            <h4 class="text-base font-bold text-gray-900 leading-7 mb-2">
                ${article.title || "Untitled article"}
            </h4>

            <p class="text-sm text-gray-600 leading-6 mb-4">
                ${article.summary || "No article summary was provided."}
            </p>

            <a
                href="${article.link || "#"}"
                target="_blank"
                rel="noopener noreferrer"
                class="inline-flex items-center gap-2 text-sm font-semibold text-ink hover:underline"
            >
                Read source
                <span aria-hidden="true">↗</span>
            </a>
        `;

        container.appendChild(card);
    });
}

function renderCoin(coin) {
    const data = analysisData[coin];
    if (!data) return;

    const newsLabel = data.news_context?.sentiment_label || "NEUTRAL";
    const articleCount = data.news_context?.article_count ?? 0;

    document.getElementById("heroCoin").textContent = coin;
    document.getElementById("heroPrice").textContent = formatPrice(data.current_price);

    document.getElementById("heroSetupScore").textContent = `${Math.round(data.setup_score ?? 0)} / 100`;
    document.getElementById("heroScoreHintShort").textContent = "This score shows how strong and clear the opportunity is";

    document.getElementById("heroSignalStrength").textContent = data.signal_strength || "-";
    document.getElementById("heroTradeStance").textContent = data.trade_stance || "-";
    document.getElementById("heroRiskPosture").textContent = data.risk_posture || data.risk_level || "LOW";
    document.getElementById("heroNewsImpact").textContent = newsLabel;
    document.getElementById("heroNewsMeta").textContent = `${articleCount} article(s)`;

    document.getElementById("heroSummary").textContent = data.ai_summary || data.insight || "No AI summary available.";
    document.getElementById("heroSignalExplainer").textContent = data.signal_explainer || "No signal explanation is available.";
    document.getElementById("heroScoreHint").textContent = data.confidence_explainer || "This score shows how strong and clear the opportunity is";

    const signalEl = document.getElementById("heroSignal");
    signalEl.textContent = data.signal || "HOLD";
    signalEl.className = `px-3 py-1.5 rounded-full border text-sm font-bold uppercase tracking-wide ${signalClass(data.signal || "HOLD")}`;

    document.getElementById("heroRegime").textContent = data.market_regime || "Transitional";

    document.getElementById("bullCase").textContent = formatPercent(data.scenario_analysis?.bull_case ?? 0);
    document.getElementById("baseCase").textContent = formatPercent(data.scenario_analysis?.base_case ?? 0);
    document.getElementById("bearCase").textContent = formatPercent(data.scenario_analysis?.bear_case ?? 0);

    document.getElementById("riskMdd").textContent = formatPercent(data.max_drawdown ?? 0);
    document.getElementById("riskVar").textContent = formatPercent(data.value_at_risk ?? 0);
    document.getElementById("riskReward").textContent = data.risk_reward === null || data.risk_reward === undefined ? "-" : formatPlain(data.risk_reward, 2);
    document.getElementById("volatilityValue").textContent = formatPlain(data.volatility ?? 0, 4);

    document.getElementById("btWinRate").textContent = formatPercent(data.bt_win_rate ?? 0, 1);
    document.getElementById("btPnl").textContent = formatPercent(data.bt_pnl ?? 0, 2);
    document.getElementById("btTrades").textContent = data.bt_trades ?? 0;

    document.getElementById("mtfShort").textContent = data.mtf_short || "-";
    document.getElementById("mtfShort").className = `text-sm font-bold ${mtfTone(data.mtf_short)}`;

    document.getElementById("mtfMed").textContent = data.mtf_med || "-";
    document.getElementById("mtfMed").className = `text-sm font-bold ${mtfTone(data.mtf_med)}`;

    document.getElementById("mtfLong").textContent = data.mtf_long || "-";
    document.getElementById("mtfLong").className = `text-sm font-bold ${mtfTone(data.mtf_long)}`;

    document.getElementById("bullishFactors").innerHTML = listToHtml(data.bullish_factors || [], "bull");
    document.getElementById("bearishFactors").innerHTML = listToHtml(data.bearish_factors || [], "bear");
    document.getElementById("watchpoints").innerHTML = listToHtml(data.watchpoints || [], "watch");

    const newsSummary = data.news_context?.summary || "No current news summary is available.";
    document.getElementById("newsSummary").textContent = newsSummary;

    const newsBadge = document.getElementById("newsLabelBadge");
    newsBadge.textContent = newsLabel;
    newsBadge.className = `px-3 py-1.5 rounded-full text-sm font-bold ${sentimentClass(newsLabel)}`;

    renderNewsArticles(data.news_context?.top_articles || []);

    document.getElementById("techRsi").textContent = formatPlain(data.technical_context?.rsi ?? 0, 2);
    document.getElementById("techAtr").textContent = formatPlain(data.technical_context?.atr ?? 0, 2);
    document.getElementById("techMacd").textContent = formatPlain(data.technical_context?.macd ?? 0, 2);
    document.getElementById("techMacdSignal").textContent = formatPlain(data.technical_context?.macd_signal ?? 0, 2);
    document.getElementById("techEma20").textContent = formatPlain(data.technical_context?.ema20 ?? 0, 2);
    document.getElementById("techSma20").textContent = formatPlain(data.technical_context?.sma20 ?? 0, 2);

    updateViewMeta();
}

function applyLiveTrade(coin, tradePayload) {
    if (!analysisData[coin]) return;

    const livePrice = parseFloat(tradePayload.p);
    analysisData[coin].current_price = livePrice;

    if (selectedCoin === coin) {
        document.getElementById("heroPrice").textContent = formatPrice(livePrice);
    }
}

function applyLiveKline(coin, streamPayload) {
    if (!analysisData[coin]) return;

    const k = streamPayload.k;
    if (!k) return;

    const candle = {
        time: Math.floor(k.t / 1000),
        open: parseFloat(k.o),
        high: parseFloat(k.h),
        low: parseFloat(k.l),
        close: parseFloat(k.c),
        volume: parseFloat(k.q),
        quote_volume: parseFloat(k.q)
    };

    const cacheKey = getRawHistoryCacheKey(coin);

    if (!rawHistoryCache[cacheKey]) {
        rawHistoryCache[cacheKey] = { candles: [] };
    }

    const series = rawHistoryCache[cacheKey].candles;
    const last = series.length ? series[series.length - 1] : null;

    if (last && last.time === candle.time) {
        series[series.length - 1] = candle;
    } else if (!last || candle.time > last.time) {
        series.push(candle);
        if (series.length > 5000) {
            series.shift();
        }
    }

    analysisData[coin].current_price = candle.close;

    if (selectedCoin === coin) {
        document.getElementById("heroPrice").textContent = formatPrice(candle.close);
    }
}

function closeLiveStream() {
    if (liveReconnectTimer) {
        clearTimeout(liveReconnectTimer);
        liveReconnectTimer = null;
    }

    if (liveSocket) {
        liveSocket.onopen = null;
        liveSocket.onmessage = null;
        liveSocket.onerror = null;
        liveSocket.onclose = null;
        liveSocket.close();
        liveSocket = null;
    }

    liveStreamCoin = null;
}

function connectLiveStream(coin) {
    if (!coin) return;

    if (liveStreamCoin === coin && liveSocket && liveSocket.readyState === WebSocket.OPEN) {
        return;
    }

    closeLiveStream();

    liveStreamCoin = coin;
    const pair = getBinancePair(coin);
    const url = `wss://stream.binance.com:9443/stream?streams=${pair}@trade/${pair}@kline_1h`;

    setConnectionStatus(`Connecting live ${coin} stream...`, "warning");

    liveSocket = new WebSocket(url);

    liveSocket.onopen = () => {
        setConnectionStatus(`Live ${coin} stream connected`, "success");
    };

    liveSocket.onmessage = (event) => {
        try {
            const payload = JSON.parse(event.data);
            const streamName = payload.stream;
            const data = payload.data;

            if (!streamName || !data) return;

            if (streamName.endsWith("@trade")) {
                applyLiveTrade(coin, data);
                return;
            }

            if (streamName.endsWith("@kline_1h")) {
                applyLiveKline(coin, data);
            }
        } catch (error) {
            console.error("Live stream parse error:", error);
        }
    };

    liveSocket.onerror = () => {
        setConnectionStatus(`Live ${coin} stream error`, "error");
    };

    liveSocket.onclose = () => {
        setConnectionStatus(`Live ${coin} stream disconnected`, "warning");

        if (selectedCoin === coin) {
            liveReconnectTimer = setTimeout(() => {
                connectLiveStream(coin);
            }, 2000);
        }
    };
}

async function fetchHistory(symbol, interval, rangeKey) {
    const url = `/api/market-history/?symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(interval)}&range=${encodeURIComponent(rangeKey)}`;
    const response = await fetch(url);

    if (!response.ok) {
        throw new Error(`Failed to fetch history (${response.status})`);
    }

    return response.json();
}

async function loadHistoryAndRenderChart(forceFetch = false) {
    if (!selectedCoin || !analysisData[selectedCoin]) return;

    const cacheKey = getRawHistoryCacheKey(selectedCoin);

    try {
        let rawHistoryPayload = rawHistoryCache[cacheKey];

        if (!rawHistoryPayload || forceFetch) {
            rawHistoryPayload = await fetchHistory(selectedCoin, "1h", selectedRange);
            rawHistoryCache[cacheKey] = rawHistoryPayload;
        }

        renderChart(rawHistoryPayload, analysisData[selectedCoin], !forceFetch);
    } catch (error) {
        console.error("Failed to load history:", error);
        setConnectionStatus("Failed to load chart history", "error");
    }
}

async function loadAnalysis() {
    try {
        const response = await fetch("/api/ai-analysis/");

        if (response.status === 202) {
            setConnectionStatus("Analysis is warming up", "warning");
            return false;
        }

        if (!response.ok) {
            throw new Error(`Failed to load analysis (${response.status})`);
        }

        const newAnalysisData = await response.json();
        const coins = Object.keys(newAnalysisData);

        if (!coins.length) {
            setConnectionStatus("No analysis data available", "error");
            return false;
        }

        analysisData = newAnalysisData;

        if (!selectedCoin || !analysisData[selectedCoin]) {
            selectedCoin = coins[0];
        }

        buildCoinSelector(coins);
        buildIntervalSelector();

        renderCoin(selectedCoin);
        connectLiveStream(selectedCoin);

        document.getElementById("loadingState").classList.add("hidden");
        document.getElementById("dashboardContent").classList.remove("hidden");

        updateLastUpdatedLabel();
        setConnectionStatus(`Analysis ready for ${selectedCoin}`, "success");
        return true;
    } catch (error) {
        console.error("Failed to load analysis:", error);
        setConnectionStatus("Failed to load analysis", "error");
        return false;
    }
}

async function refreshAnalysisPreserveSelection() {
    const previousCoin = selectedCoin;
    const previousInterval = selectedInterval;

    const ok = await loadAnalysis();
    if (!ok) return;

    if (previousCoin && analysisData[previousCoin]) {
        selectedCoin = previousCoin;
        selectedInterval = previousInterval;
        buildCoinSelector(Object.keys(analysisData));
        buildIntervalSelector();
        renderCoin(selectedCoin);

        if (!isInteractionLocked() && !isDraggingChart) {
            await loadHistoryAndRenderChart(false);
        }
    }
}

document.addEventListener("DOMContentLoaded", async () => {
    if (window.ChartZoom) {
        Chart.register(window.ChartZoom);
    }

    const resetZoomBtn = document.getElementById("resetZoomBtn");
    if (resetZoomBtn) {
        resetZoomBtn.addEventListener("click", () => {
            resetChartToDefaultView();
            lockChartInteraction(600);
        });
    }

    bindHorizontalTrackpadPan();
    bindMouseDragPan();

    const ok = await loadAnalysis();
    if (ok) {
        await loadHistoryAndRenderChart(true);
    }

    setInterval(async () => {
        await refreshAnalysisPreserveSelection();
    }, 60000);
});