let analysisData = {};
let selectedCoin = null;
let priceChart = null;
let liveSocket = null;
let liveReconnectTimer = null;
let liveStreamCoin = null;

function formatPrice(value) {
    if (value === null || value === undefined || isNaN(value)) return "-";

    if (value >= 1000) return `$${Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
    if (value >= 1) return `$${Number(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 })}`;
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

        button.addEventListener("click", () => {
            selectedCoin = coin;
            buildCoinSelector(Object.keys(analysisData));
            renderCoin(coin);
            connectLiveStream(coin);
        });

        container.appendChild(button);
    });
}

function renderChart(coin, data) {
    const ctx = document.getElementById("priceChart").getContext("2d");
    const chartData = data.chart_data || [];
    const forecast = data.forecast_next_hours || [];

    const labels = chartData.map(item => {
        const d = new Date(item.time * 1000);
        return `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
    });

    const closeValues = chartData.map(item => item.close);

    const forecastLabels = [...labels];
    for (let i = 1; i <= forecast.length; i++) {
        forecastLabels.push(`F+${i}`);
    }

    const historySeries = [...closeValues];
    while (historySeries.length < forecastLabels.length) {
        historySeries.push(null);
    }

    const forecastSeries = new Array(Math.max(closeValues.length - 1, 0)).fill(null)
        .concat(closeValues.length ? [closeValues[closeValues.length - 1]] : [null])
        .concat(forecast);

    if (priceChart) {
        priceChart.destroy();
    }

    priceChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: forecastLabels,
            datasets: [
                {
                    label: `${coin} Price`,
                    data: historySeries,
                    borderColor: "#111827",
                    backgroundColor: "rgba(17, 24, 39, 0.05)",
                    borderWidth: 2.2,
                    pointRadius: 0,
                    tension: 0.35,
                    fill: false
                },
                {
                    label: "AI Forecast",
                    data: forecastSeries,
                    borderColor: "#16a34a",
                    backgroundColor: "rgba(22, 163, 74, 0.08)",
                    borderWidth: 2.2,
                    pointRadius: 0,
                    borderDash: [7, 7],
                    tension: 0.35,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: "index",
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
                        }
                    }
                },
                tooltip: {
                    backgroundColor: "#111827",
                    titleColor: "#ffffff",
                    bodyColor: "#ffffff",
                    displayColors: false
                }
            },
            scales: {
                x: {
                    grid: {
                        color: "rgba(229, 231, 235, 0.7)"
                    },
                    ticks: {
                        color: "#6b7280",
                        maxTicksLimit: 12
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
    document.getElementById("riskReward").textContent = formatPlain(data.risk_reward ?? 0, 2);
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

    renderChart(coin, data);
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
        value: parseFloat(k.q)
    };

    analysisData[coin].current_price = candle.close;

    if (!analysisData[coin].chart_data) {
        analysisData[coin].chart_data = [];
    }

    const series = analysisData[coin].chart_data;
    const last = series.length ? series[series.length - 1] : null;

    if (last && last.time === candle.time) {
        series[series.length - 1] = candle;
    } else if (!last || candle.time > last.time) {
        series.push(candle);

        if (series.length > 100) {
            series.shift();
        }
    }

    if (selectedCoin === coin) {
        document.getElementById("heroPrice").textContent = formatPrice(candle.close);
        renderChart(coin, analysisData[coin]);
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

async function loadAnalysis() {
    try {
        const response = await fetch("/api/ai-analysis/");

        if (response.status === 202) {
            setConnectionStatus("Analysis is warming up", "warning");
            return;
        }

        if (!response.ok) {
            throw new Error("Failed to fetch analysis data.");
        }

        const data = await response.json();
        analysisData = data;
        const coins = Object.keys(analysisData);

        if (!coins.length) {
            throw new Error("No coin analysis was returned.");
        }

        if (!selectedCoin || !analysisData[selectedCoin]) {
            selectedCoin = coins[0];
        }

        buildCoinSelector(coins);
        renderCoin(selectedCoin);
        connectLiveStream(selectedCoin);

        document.getElementById("loadingState").classList.add("hidden");
        document.getElementById("errorState").classList.add("hidden");
        document.getElementById("dashboardContent").classList.remove("hidden");
    } catch (error) {
        document.getElementById("loadingState").classList.add("hidden");
        document.getElementById("dashboardContent").classList.add("hidden");
        document.getElementById("errorState").classList.remove("hidden");
        document.getElementById("errorMessage").textContent = error.message || "Something went wrong while loading analysis data.";
        setConnectionStatus("Connection issue", "error");
    }
}

loadAnalysis();
setInterval(loadAnalysis, 60000);